import pandas as pd
import numpy as np
import datetime as dt
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay, \
    USLaborDay, USThanksgivingDay
import calendar


class MSHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday(
            'New Years Day',
            month=1,
            day=1,
            observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday(
            'Independence Day',
            month=7,
            day=4,
            observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday(
            'Day After Thanksgiving',
            month=11,
            day=26,
            observance=nearest_workday),
        Holiday(
            'Christmas Eve',
            month=12,
            day=24,
            observance=nearest_workday),
        Holiday(
            'Christmas',
            month=12,
            day=25,
            observance=nearest_workday)]


# instantiate MS Holidays
inst = MSHolidayCalendar()
# build them
holidays = inst.holidays(dt.datetime(2012, 12, 31), dt.datetime(2099, 12, 31))

# month mapping
month_map = dict(enumerate(calendar.month_abbr))
# reverse it
month_map = {v: k for k, v in month_map.items()}


# create a work week mapper
def WorkWeekMapper(year, week, day):
    '''
    Build work weeks that capture dates like:
    Today's Date: 04 October 2021
    Mapper Result: 21ww10.1 ; 1 is Monday
    '''
    yy = abs(year) % 100
    week = week
    day = day
    create_date = str(yy) + 'ww' + str(week) + '.' + str(day)
    return create_date


# month number
def create_dim_date(start='2013-01-01', end='2099-12-31'):
    df = pd.DataFrame({"date": pd.date_range(start, end)})
    df['dateString'] = df['date'].dt.strftime('%Y%m%d')
    # iso; years, weeks, week days; Monday = 1
    df[['internationalYear', 'internationalWeek',
        'internationalDay']] = df['date'].dt.isocalendar().astype(int)
    # gregorian
    df['Year'] = df['date'].dt.year
    df['Day'] = df['date'].dt.day
    df['dayName'] = df['date'].dt.day_name()
    df['Day'] = df['date'].dt.day
    df['monthName'] = df['date'].dt.month_name()
    df['abbreviatedMonthName'] = df['date'].dt.month_name().str[:3]
    df['monthNumber'] = df['abbreviatedMonthName'].map(month_map)
    df['monthYear'] = df['abbreviatedMonthName'] + '-' + df['Year'].astype(str)
    df['mondayStartOfWeek'] = (df['date'] - pd.to_timedelta(
        df['date'].dt.weekday, unit='D')).dt.strftime('%Y%m%d').astype(int)
    df['americanWeek'] = df.date.dt.strftime('%W').astype(int)
    df['daysInMonth'] = df['date'].dt.days_in_month
    df['dayofYear'] = df['date'].dt.dayofyear
    df["Quarter"] = df['date'].dt.quarter
    df["isWeekDayFlag"] = df['internationalDay'].map(
        lambda day: 1 if day < 5 else 0)
    df["isWeekendFlag"] = df['internationalDay'].map(
        lambda day: 1 if day in [5, 6] else 0)
    df['fiscalYear'] = df['date'].map(
        lambda x: x.year + 1 if x.month > 6 else x.year)
    df['fiscalQuarterName'] = pd.PeriodIndex(
        df['date'], freq='Q-JUN').strftime('Q%q')
    df['fiscalQuarter'] = df['fiscalQuarterName'].map(
        {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4})
    df['fiscalYearQuarter'] = df['fiscalQuarterName'] + \
        'FY' + df['fiscalYear'].astype(str).str.slice(start=2)
    # outlook weekNum
    df['yearFirstDay'] = [x.replace(month=1, day=1) for x in df['date']]
    df['yearBaseDay'] = [x - dt.timedelta(days = x.weekday()) for x in df['yearFirstDay']]
    # Find the Monday before the first day of next year
    df['nextYearFirstDay'] = [x.replace(month = 12, day = 31) + dt.timedelta(days = 1) for x in df['date']]
    df['nextYearBaseDay'] = [x - dt.timedelta(days = x.weekday()) for x in df['nextYearFirstDay']]

    # Determine the right start day to count from for each date
    df['baseDay'] = [y if x >= y else z for (x, y, z) in zip(df['date'], df['nextYearBaseDay'], df['yearBaseDay'])]

    # Figure out the week number
    df['outlookWeek'] = [((x - y).days // 7) + 1 for (x, y) in zip(df['date'], df['baseDay'])]
    # drop excess cols
    df.drop(['yearFirstDay', 'yearBaseDay', 'nextYearFirstDay', 'nextYearBaseDay',
             'baseDay'], axis=1, inplace=True)
    # put date key up front
    df.insert(0, 'dateKey', (df['date'].dt.strftime('%Y%m%d').astype(int)))
    return df


# create dim date
df = create_dim_date()

# generate work weeks
df['internationalWorkWeek'] = [WorkWeekMapper(*a) for a in tuple(
    zip(df["internationalYear"],
        df["internationalWeek"],
        df["internationalDay"])
)]
df['americanWorkWeek'] = [WorkWeekMapper(*a) for a in tuple(
    zip(df["Year"],
        df["americanWeek"],
        df["internationalDay"])
)]

# map holidays
df['isHolidayFlag'] = np.where(df['date'].isin(holidays), 1, 0)

# date not yet occurred -- date key 0
rep1 = np.array([0,
                 pd.Timestamp('2099-12-31 00:00:00'),
                 '<not yet occurred>',
                 2099,
                 53,
                 4,
                 2099,
                 31,
                 'Thursday',
                 'January',
                 'Jan',
                 12,
                 'Jan-2099',
                 20991228,
                 52,
                 31,
                 365,
                 4,
                 1,
                 0,
                 2100,
                 'Q2',
                 2,
                 'Q2FY00',
                 1,
                 '99ww53.4',
                 '99ww52.4',
                 0])


df = df.append(pd.DataFrame(rep1.reshape(-1, 28),
               columns=df.columns.values), ignore_index=True)

# missing dates -- date key -9999
rep2 = np.array([-9999,
                 pd.Timestamp('2099-12-31 00:00:00'),
                 '<missing>',
                 2099,
                 53,
                 4,
                 2099,
                 31,
                 'Thursday',
                 'January',
                 'Jan',
                 12,
                 'Jan-2099',
                 20991228,
                 52,
                 31,
                 365,
                 4,
                 1,
                 0,
                 2100,
                 'Q2',
                 2,
                 'Q2FY00',
                 1,
                 '99ww53.4',
                 '99ww52.4',
                 0])

df = df.append(pd.DataFrame(rep2.reshape(-1, 28),
               columns=df.columns.values), ignore_index=True)

# create some ordering col integers
df['fiscalYearQuarterOrder'] = pd.factorize(df['fiscalYearQuarter'])[0]
df['monthYearOrder'] = pd.factorize(df['monthYear'])[0]
