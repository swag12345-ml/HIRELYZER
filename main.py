import streamlit as st

st.set_page_config(page_title="Cinematic Landing", layout="wide")

# Hide default Streamlit elements
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Cinematic Landing Page
html_code = """
<style>
body {
  margin: 0;
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
  overflow: hidden;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Container */
.scene {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

/* Walking man */
.man {
  position: absolute;
  bottom: 40px;
  left: -150px;
  width: 120px;
  height: 240px;
  background: url('https://cdn-icons-png.flaticon.com/512/860/860531.png') no-repeat center/contain;
  animation: walkAcross 6s forwards;
}

/* Bag */
.bag {
  position: absolute;
  bottom: 80px;
  left: 120px;
  width: 60px;
  height: 60px;
  background: url('https://cdn-icons-png.flaticon.com/512/3135/3135715.png') no-repeat center/contain;
  animation: throwBag 6s forwards;
}

/* Form box */
.form-box {
  position: absolute;
  bottom: 100px;
  left: 50%;
  transform: translateX(-50%) scale(0);
  width: 320px;
  padding: 30px;
  background: rgba(255,255,255,0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  color: white;
  text-align: center;
  animation: showForm 8s forwards;
}

.form-box h2 {
  margin: 0 0 20px;
  font-size: 24px;
}

.form-box input {
  width: 100%;
  padding: 12px;
  margin: 8px 0;
  border: none;
  border-radius: 8px;
}

.form-box button {
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 8px;
  background: #5cf2d6;
  font-weight: bold;
  cursor: pointer;
  margin-top: 10px;
}

.form-box button:hover {
  background: #46c4ab;
}

/* Animations */
@keyframes walkAcross {
  0% { left: -150px; }
  50% { left: 200px; }
  100% { left: 180px; transform: rotate(-5deg); } /* lean on form */
}

@keyframes throwBag {
  0% { bottom: 80px; left: 120px; }
  40% { bottom: 300px; left: 200px; }
  60% { bottom: 250px; left: 240px; }
  100% { bottom: 150px; left: 50%; transform: scale(0); }
}

@keyframes showForm {
  0% { transform: translateX(-50%) scale(0); opacity: 0; }
  80% { transform: translateX(-50%) scale(1.05); opacity: 1; }
  100% { transform: translateX(-50%) scale(1); opacity: 1; }
}
</style>

<div class="scene">
  <div class="man"></div>
  <div class="bag"></div>
  <div class="form-box">
    <h2>Welcome</h2>
    <input type="text" placeholder="Username">
    <input type="password" placeholder="Password">
    <button>Login</button>
  </div>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)
