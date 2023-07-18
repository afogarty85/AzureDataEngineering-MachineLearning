FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest
USER root

# install FreeTDS and dependencies
RUN apt-get update \
 && apt-get install unixodbc -y \
 && apt-get install curl -y \
 && apt-get install unixodbc-dev -y \
 && apt-get install freetds-dev -y \
 && apt-get install freetds-bin -y \
 && apt-get install tdsodbc -y \
 && apt-get install --reinstall build-essential -y

# populate "ocbcinst.ini"
RUN echo "[FreeTDS]\n\
Description = FreeTDS unixODBC Driver\n\
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

# Add SQL Server ODBC Driver 17 for Ubuntu 18.04
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated msodbcsql17
RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc

# install pyodbc (and, optionally, sqlalchemy)
RUN pip install --trusted-host pypi.org pyodbc
RUN pip install --trusted-host pypi.org pandas==1.5.3
RUN pip install --trusted-host pypi.org azure-storage-file-share
RUN pip install --trusted-host pypi.org azure-storage-file-datalake
RUN pip install --trusted-host pypi.org python-dotenv
RUN pip install --trusted-host pypi.org azure-keyvault-secrets
RUN pip install --trusted-host pypi.org azure-identity
RUN pip install --trusted-host pypi.org nest_asyncio
RUN pip install --trusted-host pypi.org aiohttp
RUN pip install --trusted-host pypi.org numpy
RUN pip install --trusted-host pypi.org tenacity
RUN pip install --trusted-host pypi.org pathlib
RUN pip install --trusted-host pypi.org loguru
RUN pip install --trusted-host pypi.org adal
RUN pip install --trusted-host pypi.org azureml-core
RUN pip install --trusted-host pypi.org pyarrow
RUN pip install --trusted-host pypi.org fastparquet
RUN pip install --trusted-host pypi.org openpyxl
RUN pip install --trusted-host pypi.org azure-kusto-data
RUN pip install --trusted-host pypi.org azure-kusto-ingest
RUN pip install --trusted-host pypi.org lxml
RUN pip install --trusted-host pypi.org azureml-dataset-runtime
RUN pip install --trusted-host pypi.org fuzzywuzzy