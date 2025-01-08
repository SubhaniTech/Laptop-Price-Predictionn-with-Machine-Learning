import streamlit as st
import pickle
import numpy as np
import platform
import psutil
import socket
from datetime import datetime

# Function to get system information
def get_system_info():
    system_info = {}
    system_info["Processor"] = platform.processor()

    # OS information
    system_info["OS"] = platform.system() + " " + platform.release()
    system_info["Arch"] = platform.architecture()[0]

    # CPU information
    system_info["CPU Cores"] = psutil.cpu_count(logical=False)
    system_info["Logical CPUs"] = psutil.cpu_count(logical=True)
    system_info["CPU Usage (%)"] = psutil.cpu_percent(interval=1)

    # Memory Information
    system_info["Total Memory (GB)"] = psutil.virtual_memory().total / (1024 ** 3)
    system_info["Available Memory (GB)"] = psutil.virtual_memory().available / (1024 ** 3)
    system_info["Used Memory (GB)"] = psutil.virtual_memory().used / (1024 ** 3)
    system_info["Memory Usage (%)"] = psutil.virtual_memory().percent

    # Disk Information
    disk_usage = psutil.disk_usage('/')
    system_info["Total Disk Space (GB)"] = disk_usage.total / (1024 ** 3)
    system_info["Used Disk Space (GB)"] = disk_usage.used / (1024 ** 3)
    system_info["Free Disk Space (GB)"] = disk_usage.free / (1024 ** 3)
    system_info["Disk Usage (%)"] = disk_usage.percent

    # Network Information
    system_info["Host Name"] = socket.gethostname()
    system_info["IP Address"] = socket.gethostbyname(socket.gethostname())

    # Boot Time
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    system_info["Boot Time"] = boot_time.strftime("%Y-%m-%d %H:%M:%S")

    return system_info
# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# Display system information
st.subheader("System Information")
system_info = get_system_info()
for key, value in system_info.items():
    st.write(f"**{key}**: {value}")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('Cpu',df['CPU Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price = " + str(int(np.exp(pipe.predict(query)[0]))))

