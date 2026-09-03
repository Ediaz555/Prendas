import os

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras


st.set_page_config(page_title="medidior de prendas", page_icon="👕")
st.title("medidior de prendas")
st.write("Dibuja una prenda o sube una imagen para clasificarla.")

CLASS_NAMES = [
	"Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo",
	"Sandalia", "Camisa", "Zapatos deportiva", "Bolso", "Botas",
]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prendas.keras")


@st.cache_resource
def load_model():
	return keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(image):
	image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
	return np.asarray(image, dtype=np.float32) / 255.0


def predict(image):
	pixels = prepare_image(image)
	model = load_model()
	input_shape = model.input_shape
	if isinstance(input_shape, list):
		input_shape = input_shape[0]
	if len(input_shape) == 4:
		batch = pixels[np.newaxis, ..., np.newaxis]
	else:
		batch = pixels[np.newaxis, ...]
	probabilities = np.asarray(model.predict(batch, verbose=0))[0]
	index = int(np.argmax(probabilities))
	return index, probabilities


option = st.radio("Selecciona una opción", ["Dibujar", "Subir imagen"], horizontal=True)
image = None

if option == "Dibujar":
	canvas = st_canvas(
		fill_color="rgba(255, 255, 255, 1)",
		stroke_width=12,
		stroke_color="#FFFFFF",
		background_color="#000000",
		width=280,
		height=280,
		drawing_mode="freedraw",
		key="prenda_canvas",
	)
	if canvas.image_data is not None and np.any(canvas.image_data[:, :, 3] > 0):
		image = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
else:
	uploaded = st.file_uploader(
		"Carga una imagen", type=["png", "jpg", "jpeg", "bmp"]
	)
	if uploaded is not None:
		image = Image.open(uploaded)
		st.image(image, caption="Imagen cargada", width=280)

if image is not None and st.button("Predecir"):
	try:
		index, probabilities = predict(image)
		st.success(f"Predicción: {CLASS_NAMES[index]}")
		st.write(f"Confianza: {probabilities[index] * 100:.2f}%")
		st.bar_chart(
			{name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)}
		)
	except FileNotFoundError:
		st.error("No se encontró Prendas.keras en la carpeta de la aplicación.")
	except Exception as error:
		st.error(f"No fue posible realizar la predicción: {error}")

st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	- Dibuja con el lápiz de ancho medio sobre el fondo negro, o carga una imagen.
	- La imagen se convierte a escala de grises, se cambia a 28 x 28 píxeles y se normaliza dividiendo entre 255.
	- Para mejores resultados, usa imágenes similares a las utilizadas para entrenar el modelo: fondo negro y prenda clara.
	- El modelo utiliza una salida softmax para calcular las probabilidades de las diez clases.
	"""
)
