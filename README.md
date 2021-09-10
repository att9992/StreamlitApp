Annually Cumulative Oil Production App
==========================


## Create a Virtual Environment

After cloning or downloading the repo, create a Python virtual environment with:

```
python -m venv .env
```

This will create the virtual environment in the project directory as `.env
## Activate the Virtual Environment

Now activate the virtual environment. on macOS, Linux and Unix systems, use:

```
source .env/bin/activate
```

On Windows with `cmd.exe`:

```
.env\Scripts\activate.bat
```

Or Windows with PowerShell:

```
.\.env\Scripts\activate.ps1
```

## Install dependencies

Install streamlit library

```
pip install streamlit
```

Install scikit-learn library 

```
pip install scikit-learn
```

Install plotly library 

```
pip install plotly
```

Install lightgbm library 

```
pip install lightgbm
```
## Run web app locally 

```
streamlit run app.py
```
