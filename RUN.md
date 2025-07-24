# Instructions for running locally on Windows

## Install Python 3.11

Go to https://apps.microsoft.com/detail/9NRWMJP3717K?hl=en-us&gl=US&ocid=pdpshare and install Python 3.11 on your computer.

*Note: You must install this exact version of Python, or you won't be able to run the app..*

## Download Source Code

1. Go to https://github.com/JaredWinkens/CrashAnalytics

2. Click the green "Code" button

3. Click "Download ZIP"

4. Extract the zip file into your home directory, e.g., `C:\Users\<your_name>`


## Download Data

1. Go to https://sunypoly-my.sharepoint.com/:u:/g/personal/winkenj_sunypoly_edu/EaP9kaLP0_9BvByH6s21d2gB-x4Av16vk1uAShGw3beIoA?e=lyomjK

2. Download the zip file

3. Extract the zip file into the source code directory, e.g., `C:\Users\<your_name>\CrashAnalytics-main`

## Run the App

Open **command prompt** and execute the following commands

### Step 1 - Create a virtual environment

```
python3.11 -m venv <environment_name>
```

### Step 2 - Activate virtual environment

```
<environment_name>\Scripts\activate.bat
```

### Step 3 - Move to the directory containing the source code

```
cd CrashAnalytics-main
```

### Step 4 - Install requirements
*Note: This may take some time.*
```
pip install -r requirements.txt
```

### Step 5 - Create configuration file

```
python create_config_file.py
```

### Step 6 - Run the app

```
python app.py
```

### Step 7 - View on the web

The app should be running on `http://127.0.0.1:8080`. Copy and paste this URL into the browser, and you should see the web app running.
