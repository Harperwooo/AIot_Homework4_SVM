
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC


# Streamlit App
st.title('3D SVM with streamlit')
st.text("使用滑桿調整分類的距離閾值，可以即時顯示3D散點圖和超平面")

# Sidebar for adjusting semi-major and semi-minor axis
semi_major_axis = st.slider('Semi-Major Axis', min_value=1.0, max_value=10.0, value=5.0, step=0.1)
semi_minor_axis = st.slider('Semi-Minor Axis', min_value=1.0, max_value=10.0, value=3.0, step=0.1)

# Generate data and perform calculations
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from the origin
distances = np.sqrt((x1 / semi_major_axis)**2 + (x2 / semi_minor_axis)**2)

# Assign labels Y=0 for points within distance 1, Y=1 for the rest
Y = np.where(distances < 1, 0, 1)

def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[Y==0], x2[Y==0], x3[Y==0], c='purple', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], x3[Y==1], c='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color')
ax.legend()

# Create a meshgrid to plot the separating hyperplane
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Display the plot using Streamlit
st.pyplot(fig)









#How to run streamlit---run the code below in terminal
##streamlit run streamlit程式名稱
#streamlit run streamlit_app.py