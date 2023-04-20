import uuid
import msal
from flask_session import Session
from flask import Flask, render_template, redirect, url_for, session, request
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms.fields import DecimalRangeField, SubmitField, StringField, BooleanField, SelectField, IntegerField
from wtforms.validators import Length, InputRequired, DataRequired
import numpy as np
import pandas as pd
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts import concatenate
from darts import TimeSeries
from darts.utils.losses import SmapeLoss, MAELoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae, mase, rmse
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.models import *
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import itertools
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, mean_pinball_loss
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# init flask
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = uuid.uuid4().hex
sess = Session()
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


# MSAL config
CLIENT_ID = os.environ["AZURE_CLIENT_ID"]  # (client) ID of app registration / service principal
CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]  # client secret for SP above

# hard code for local flask testing; remove when publishing! -- otherwise attach the above to the configuration blade in the web app
# and setup authentication tab below it for AAD restricted access!
# CLIENT_ID = ''
# CLIENT_SECRET = ''

AUTHORITY = "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47"

REDIRECT_PATH = "/getAToken"  # Used for forming an absolute URL to your redirect URI.
                              # The absolute URL must match the redirect URI you set
                              # in the app's registration in the Azure portal.

ENDPOINT = 'https://graph.microsoft.com/v1.0/users'  # This resource requires no admin consent

# https://docs.microsoft.com/en-us/graph/permissions-reference
SCOPE = ["User.ReadBasic.All"]

# Specifies the token cache should be stored in server-side session
SESSION_TYPE = "filesystem" 


@app.before_first_request
def startup():
    global tseries
    global sess
    tseries = pd.read_csv(r'electricity_dataset.csv')
    # add 1
    tseries['kW'] = tseries['kW'] + 1
    
# form to collect user input:
class InputForm(FlaskForm):

    # code selections
    selections = {'customer1': 0, 'customer2': 1, 'customer3': 2, }

    # choose customer
    f_customerName = SelectField('Customer Name', choices=selections.keys(), validators=[InputRequired()])

    # choose horizon
    f_horizon = IntegerField('Prediction Horizon', validators=[InputRequired()])
    
    # submit button
    submit = SubmitField('Submit')


@app.route("/userinput", methods=['POST', 'GET'])
def userinput():

    form = InputForm()
    if form.validate_on_submit():

        # attach to session
        session['customerName'] = form.f_customerName.data
        session['Horizon'] = form.f_horizon.data

        return redirect(url_for('predict'))

    return render_template("index.html", form=form)


@app.route('/predict', methods = ['POST', 'GET'])
def predict():

    # feature mapper
    customer_mapper = {'customer1': 0, 'customer2': 1, 'customer3': 2, }

    # get session selection
    customerName = session['customerName']
    horizon = session['Horizon']

    # generate DF
    user_selection = pd.DataFrame({
                            'customerName': [customer_mapper.get(customerName)],
                            'Horizon': [horizon],
                        })

    print('Preparing data')
    # build time series df
    series_multi = TimeSeries.from_group_dataframe(
        tseries,
        time_col="Date",
        group_cols="Customer",
        value_cols=["kW"],
        freq='1H',
        fill_missing_dates=True
    )

    # set f32 for way faster training
    series_multi = [s.astype(np.float32) for s in tqdm(series_multi)]

    # Build train set
    train_set = [s for s in series_multi]

    # static trnsformer to transform string ID to numeric
    static_transformer = StaticCovariatesTransformer()
    train_set = static_transformer.fit_transform(train_set)

    # extract user selected customer
    user_selection_idx = user_selection['customerName'].item()
    print(f'User selection idx: {user_selection_idx}')

    # setup reverse code for key value
    key_mapping = {v: k for k, v in customer_mapper.items()}

    # gen scaler for time series
    scaler = Scaler(BoxCoxTransformer(method='guerrero', sp=2))

    # get boxcox for series selected
    train_set_transformed = scaler.fit_transform(train_set[user_selection_idx])

    # default model selection
    m = StatsForecastAutoETS()

    # fit
    m.fit(series=train_set_transformed)
    
    # get preds
    if m._is_probabilistic():
        preds = m.predict(horizon, num_samples=1000)
    else:
        preds = m.predict(horizon, num_samples=1)

    # inverse transform preds
    preds_inv = scaler.inverse_transform(preds)

    # gen plot
    fig, axs = plt.subplots(1,  figsize=(12, 8))
    train_set[user_selection_idx][-horizon:].plot(ax=axs, label='Historical', color='green')
    preds_inv.plot(ax=axs, low_quantile=0.25, high_quantile=0.75, label='25th-75th Quantile Forecast', color='blue')

    # set to df
    q_df = preds_inv.map(lambda x: np.nan_to_num(x)).quantiles_df(quantiles=([0.25, 0.5, 0.75, 0.95])).reset_index()

    # cols
    q_df.columns = ['Date', '25thQuantile', 'MedianQuantile', '75thQuantile', '95thQuantile']

    # set int
    q_df[['25thQuantile', 'MedianQuantile', '75thQuantile', '95thQuantile']] = q_df[['25thQuantile', 'MedianQuantile', '75thQuantile', '95thQuantile']].fillna(0).astype(int)

  
    new_graph_name = "graph" + ".png"

    for filename in os.listdir('static/'):
        if filename.startswith('graph_'):  # not to remove other images
            os.remove('static/' + filename)

    fig.savefig('static/' + new_graph_name)

    return render_template("image.html", graph=new_graph_name, column_names=q_df.columns.values, row_data=list(q_df.values.tolist()), zip=zip, series=key_mapping.get(user_selection_idx))



@app.route("/")
def index():
    if not session.get("user"):
        return redirect(url_for("login"))
    print(session['user'])
    return render_template('index.html', user=session["user"], version=msal.__version__)

@app.route("/login")
def login():
    # Technically we could use empty list [] as scopes to do just sign in,
    # here we choose to also collect end user consent upfront
    session["flow"] = _build_auth_code_flow(scopes=SCOPE)
    return render_template("login.html", auth_url=session["flow"]["auth_uri"], version=msal.__version__)

@app.route(REDIRECT_PATH)  # Its absolute URL must match your app's redirect_uri set in AAD
def authorized():
    try:
        cache = _load_cache()
        result = _build_msal_app(cache=cache).acquire_token_by_auth_code_flow(
            session.get("flow", {}), request.args)
        if "error" in result:
            return render_template("auth_error.html", result=result)
        session["user"] = result.get("id_token_claims")
        _save_cache(cache)
    except ValueError:  # Usually caused by CSRF
        pass  # Simply ignore them
    return redirect(url_for("userinput"))

@app.route("/logout")
def logout():
    session.clear()  # Wipe out user and its token cache from session
    return redirect(  # Also logout from your tenant's web session
        AUTHORITY + "/oauth2/v2.0/logout" +
        "?post_logout_redirect_uri=" + url_for("index", _external=True))

@app.route("/graphcall")
def graphcall():
    token = _get_token_from_cache(SCOPE)
    if not token:
        return redirect(url_for("login"))
    graph_data = requests.get(  # Use token to call downstream service
        ENDPOINT,
        headers={'Authorization': 'Bearer ' + token['access_token']},
        ).json()
    return render_template('display.html', result=graph_data)


def _load_cache():
    cache = msal.SerializableTokenCache()
    if session.get("token_cache"):
        cache.deserialize(session["token_cache"])
    return cache

def _save_cache(cache):
    if cache.has_state_changed:
        session["token_cache"] = cache.serialize()

def _build_msal_app(cache=None, authority=None):
    return msal.ConfidentialClientApplication(
        CLIENT_ID, authority=authority or AUTHORITY,
        client_credential=CLIENT_SECRET, token_cache=cache)

def _build_auth_code_flow(authority=None, scopes=None):
    return _build_msal_app(authority=authority).initiate_auth_code_flow(
        scopes or [],
        redirect_uri=url_for("authorized", _external=True))

def _get_token_from_cache(scope=None):
    cache = _load_cache()  # This web app maintains one cache per session
    cca = _build_msal_app(cache=cache)
    accounts = cca.get_accounts()
    if accounts:  # So all account(s) belong to the current signed-in user
        result = cca.acquire_token_silent(scope, account=accounts[0])
        _save_cache(cache)
        return result

app.jinja_env.globals.update(_build_auth_code_flow=_build_auth_code_flow)  # Used in template

# for https local dev; run:
# mkcert localhost 127.0.0.1 ::1
# then run:
# flask run --cert=localhost+2.pem --key=localhost+2-key.pem --port 8000

# launch
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
