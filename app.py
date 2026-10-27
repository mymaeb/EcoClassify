from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np, os, tempfile

app = Flask(__name__)
model = load_model('modelo_basura_alta_precision.keras')

with open('classes.txt', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f if line.strip()]

# --- Contramedidas por categoría ---
# Ajusta las claves para que coincidan EXACTAMENTE con tu classes.txt
INFO = {
    "organico": {
        "emoji": "🥬", "color": "#8bc34a",
        "donde": "Contenedor verde / orgánico",
        "tips": [
            "Separa restos de comida sin bolsa ni empaque.",
            "Si puedes, composta en casa: en 3 meses tienes abono.",
            "Escurre líquidos antes de tirarlo para evitar malos olores."
        ],
        "impacto": "La basura orgánica representa hasta el 50% de los residuos domésticos. Compostarla reduce el metano de los rellenos sanitarios."
    },
    "plastico": {
        "emoji": "🧴", "color": "#03a9f4",
        "donde": "Contenedor amarillo / reciclable",
        "tips": [
            "Enjuaga el envase: el plástico sucio no se recicla.",
            "Aplasta botellas y quita las tapas (van por separado).",
            "Revisa el número: el PET (1) y HDPE (2) son los más reciclables."
        ],
        "impacto": "Una botella de PET tarda hasta 450 años en degradarse. Reciclar una ahorra energía equivalente a 3 horas de una bombilla LED."
    },
    "vidrio": {
        "emoji": "🍾", "color": "#00bcd4",
        "donde": "Contenedor verde / vidrio",
        "tips": [
            "Retira tapas y corchos antes de depositarlo.",
            "No lo mezcles con cerámica ni espejos: contaminan el lote.",
            "El vidrio se recicla infinitas veces sin perder calidad."
        ],
        "impacto": "Reciclar vidrio ahorra 30% de energía frente a fabricarlo desde materia prima."
    },
    "papel": {
        "emoji": "📄", "color": "#ffc107",
        "donde": "Contenedor azul / papel",
        "tips": [
            "Que esté seco y sin grasa: el papel de cocina sucio no sirve.",
            "Quita grapas, clips y plásticos.",
            "Dóblalo para ocupar menos espacio."
        ],
        "impacto": "Reciclar 1 tonelada de papel salva cerca de 17 árboles."
    },
    "metal": {
        "emoji": "🥫", "color": "#9e9e9e",
        "donde": "Contenedor amarillo / reciclable",
        "tips": [
            "Enjuaga latas para evitar plagas y olores.",
            "Aplasta las latas para reducir volumen.",
            "El aluminio es el material más valioso del reciclaje doméstico."
        ],
        "impacto": "Reciclar aluminio usa 95% menos energía que producirlo desde bauxita."
    },
    "carton": {
        "emoji": "📦", "color": "#bf8b5e",
        "donde": "Contenedor azul / cartón",
        "tips": [
            "Desarma las cajas y quita la cinta adhesiva.",
            "Mantenlo seco: el cartón mojado pierde valor.",
            "Si está engrasado (caja de pizza), va a orgánico, no a reciclaje."
        ],
        "impacto": "El cartón reciclado reduce el uso de agua en la fabricación hasta 50%."
    },
    "textiles": {
        "emoji": "👕", "color": "#e91e63",
        "donde": "Punto de acopio textil o donación",
        "tips": [
            "Si está en buen estado, dónalo antes de tirarlo.",
            "Ropa muy desgastada puede ir a trapos o relleno industrial.",
            "Las fibras sintéticas tardan más de 200 años en degradarse."
        ],
        "impacto": "La industria textil es una de las más contaminantes: reutilizar una prenda ahorra hasta 2,700 litros de agua."
    },
}

def info_de(clase):
    clave = clase.strip().lower()
    for k, v in INFO.items():
        if k in clave:
            return v
    return {
        "emoji": "♻️", "color": "#4caf50",
        "donde": "Verifica el contenedor según tu municipio",
        "tips": ["Revisa el etiquetado del producto.", "Consulta el reglamento local de residuos."],
        "impacto": "Separar correctamente mejora el aprovechamiento de todos los materiales."
    }

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EcoClassify · Clasificador de Residuos</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    :root {
      --verde: #2e7d32; --verde-claro: #4caf50;
      --fondo: #f4f9f4; --texto: #16321a; --gris: #6b7c6e;
    }
    body {
      font-family: 'Segoe UI', system-ui, Arial, sans-serif;
      background: var(--fondo); color: var(--texto);
      min-height: 100vh; line-height: 1.5;
    }

    /* Header con z-index bajo para que las tarjetas queden encima */
    header {
      background: linear-gradient(135deg, #1b5e20, #43a047 60%, #66bb6a);
      color: #fff; text-align: center;
      padding: 3rem 1rem 6rem;   /* antes: 4rem → ahora 6rem */
      position: relative; overflow: hidden;
      z-index: 1;
    }
    header::after {
      content: ""; position: absolute; inset: 0;
      background: radial-gradient(circle at 20% 120%, rgba(255,255,255,.18), transparent 60%);
      pointer-events: none;
    }
    header h1, header p { position: relative; z-index: 2; }
    header h1 { font-size: clamp(1.6rem, 5vw, 2.4rem); letter-spacing: -.5px; }
    header p { margin-top: .6rem; opacity: .92; font-size: 1rem; }

    /* Contenedor por encima del header: ya no tapa el título */
    .container {
      max-width: 760px;
      margin: -3.5rem auto 3rem;  /* antes: -2rem → ahora -3.5rem */
      padding: 0 1rem;
      position: relative;
      z-index: 2;
    }


    .card {
      background: #fff; border-radius: 16px; padding: 1.6rem;
      margin-bottom: 1.2rem; box-shadow: 0 4px 20px rgba(20,60,25,.08);
      border: 1px solid rgba(76,175,80,.12);
    }

    h2 {
      color: var(--verde); font-size: 1.05rem; margin-bottom: 1rem;
      display: flex; align-items: center; gap: .5rem;
      text-transform: uppercase; letter-spacing: .5px;
      line-height: 1.4;
    }

    .cats { display: flex; flex-wrap: wrap; gap: .5rem; }
    .cat {
      background: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32;
      padding: .4rem .9rem; border-radius: 999px; font-size: .85rem;
      font-weight: 600; transition: transform .15s;
    }
    .cat:hover { transform: translateY(-2px); }

    ol { padding-left: 1.2rem; }
    ol li { margin-bottom: .4rem; }

    .drop {
      border: 2px dashed #a5d6a7; border-radius: 14px;
      padding: 2.2rem 1rem; text-align: center;
      background: #fafffa; transition: all .2s; cursor: pointer;
    }
    .drop:hover, .drop.drag { border-color: var(--verde-claro); background: #f1fbf1; }
    .drop p { color: var(--gris); font-size: .9rem; margin-top: .5rem; }
    #fileInput { display: none; }

    .btn {
      background: linear-gradient(135deg, #2e7d32, #43a047);
      color: #fff; border: none; padding: .85rem 1.8rem;
      border-radius: 10px; font-size: 1rem; font-weight: 600;
      cursor: pointer; box-shadow: 0 4px 12px rgba(46,125,50,.3);
      transition: transform .15s, box-shadow .15s;
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(46,125,50,.4); }
    .btn:active { transform: translateY(0); }

    #preview {
      margin: 1.2rem auto 0; max-width: 260px; width: 100%;
      display: none; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
    }
    #loading { margin-top: 1rem; color: var(--gris); display: none; }
    .spinner {
      width: 22px; height: 22px; border: 3px solid #c8e6c9;
      border-top-color: var(--verde); border-radius: 50%;
      animation: spin .8s linear infinite; display: inline-block;
      vertical-align: middle; margin-right: .5rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Resultado */
    #resultado { display: none; }
    .res-head {
      display: flex; align-items: center; gap: 1rem;
      padding-bottom: 1rem; border-bottom: 1px solid #eef4ee; margin-bottom: 1rem;
    }
    .res-emoji { font-size: 2.6rem; line-height: 1; }
    .res-clase { font-size: 1.5rem; font-weight: 700; }
    .res-sub { color: var(--gris); font-size: .9rem; }

    .barra { height: 10px; background: #e8f0e8; border-radius: 999px; overflow: hidden; margin-top: .6rem; }
    .barra-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }

    .bloque { margin-top: 1.2rem; }
    .bloque h3 {
      font-size: .8rem; text-transform: uppercase; letter-spacing: .8px;
      color: var(--gris); margin-bottom: .6rem;
    }
    .donde {
      display: inline-flex; align-items: center; gap: .5rem;
      background: #e8f5e9; color: #1b5e20; font-weight: 600;
      padding: .6rem 1rem; border-radius: 10px; font-size: .95rem;
    }
    .tips { list-style: none; }
    .tips li {
      padding: .5rem 0 .5rem 1.6rem; position: relative;
      border-bottom: 1px solid #f2f7f2; font-size: .93rem;
    }
    .tips li:last-child { border-bottom: none; }
    .tips li::before {
      content: "✓"; position: absolute; left: 0; top: .5rem;
      color: var(--verde-claro); font-weight: 700;
    }
    .impacto {
      background: #f1f8e9; border-left: 4px solid #8bc34a;
      padding: .9rem 1rem; border-radius: 0 10px 10px 0;
      font-size: .9rem; color: #33691e;
    }
    .alts { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
    .alt {
      background: #f5f5f5; border-radius: 8px; padding: .35rem .7rem;
      font-size: .8rem; color: #555;
    }
    footer { text-align: center; color: var(--gris); font-size: .8rem; padding-bottom: 2rem; }
  </style>
</head>
<body>
  <header>
    <h1>♻️ EcoClassify</h1>
    <p>Sube una foto y te decimos dónde va y cómo aprovecharlo</p>
  </header>

  <div class="container">
    <div class="card">
      <h2>🗂️ Categorías</h2>
      <div class="cats">
        <span class="cat">🥬 Orgánico</span>
        <span class="cat">🧴 Plástico</span>
        <span class="cat">🍾 Vidrio</span>
        <span class="cat">📄 Papel</span>
        <span class="cat">🥫 Metal</span>
        <span class="cat">📦 Cartón</span>
        <span class="cat">👕 Textiles</span>
      </div>
    </div>

    <div class="card">
      <h2>📷 Clasificar residuo</h2>
      <div class="drop" id="drop">
        <button class="btn" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">
          Subir archivo
        </button>
        <p>o arrastra una imagen aquí</p>
      </div>
      <input type="file" id="fileInput" accept="image/*">
      <div style="text-align:center">
        <img id="preview" alt="Vista previa">
        <div id="loading"><span class="spinner"></span>Analizando imagen...</div>
      </div>
    </div>

    <div class="card" id="resultado">
      <div class="res-head">
        <div class="res-emoji" id="r-emoji">♻️</div>
        <div>
          <div class="res-clase" id="r-clase">—</div>
          <div class="res-sub" id="r-conf">—</div>
        </div>
      </div>
      <div class="barra"><div class="barra-fill" id="r-barra" style="width:0%"></div></div>

      <div class="bloque" id="b-donde">
        <h3>📍 Dónde tirarlo</h3>
        <div class="donde" id="r-donde">—</div>
      </div>

      <div class="bloque" id="b-tips">
        <h3>💡 Tips</h3>
        <ul class="tips" id="r-tips"></ul>
      </div>

      <div class="bloque" id="b-impacto">
        <h3>🌍 Impacto</h3>
        <div class="impacto" id="r-impacto">—</div>
      </div>

      <div class="bloque" id="b-alts">
        <h3>Otras posibilidades</h3>
        <div class="alts" id="r-alts"></div>
      </div>
    </div>

    <footer>EcoClassify · clasificación automática de residuos</footer>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const drop = document.getElementById('drop');

    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
    drop.addEventListener('drop', e => {
      e.preventDefault(); drop.classList.remove('drag');
      if (e.dataTransfer.files.length) enviar(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => { if (e.target.files[0]) enviar(e.target.files[0]); });

    async function enviar(file) {
      const img = document.getElementById('preview');
      img.src = URL.createObjectURL(file);
      img.style.display = 'block';
      document.getElementById('resultado').style.display = 'none';
      document.getElementById('loading').style.display = 'block';

      const form = new FormData();
      form.append('imagen', file);

      try {
        const res = await fetch('/clasificar', { method: 'POST', body: form });
        const data = await res.json();
        document.getElementById('loading').style.display = 'none';

        if (data.error) { alert('Error: ' + data.error); return; }
        pintar(data);
      } catch (err) {
        document.getElementById('loading').style.display = 'none';
        alert('No se pudo conectar con el servidor.');
      }
    }

    function pintar(d) {
      const info = d.info || {};
      document.getElementById('resultado').style.display = 'block';
      document.getElementById('r-emoji').textContent = info.emoji || '♻️';
      document.getElementById('r-clase').textContent = d.clase;
      document.getElementById('r-conf').textContent =
        'Confianza: ' + (d.confianza * 100).toFixed(1) + '%';

      const barra = document.getElementById('r-barra');
      barra.style.width = (d.confianza * 100).toFixed(1) + '%';
      barra.style.background = info.color || '#4caf50';

      document.getElementById('r-donde').textContent = info.donde || '—';

      const ul = document.getElementById('r-tips');
      ul.innerHTML = '';
      (info.tips || []).forEach(t => {
        const li = document.createElement('li'); li.textContent = t; ul.appendChild(li);
      });

      document.getElementById('r-impacto').textContent = info.impacto || '—';

      const alts = document.getElementById('r-alts');
      alts.innerHTML = '';
      (d.alternativas || []).forEach(a => {
        const s = document.createElement('span');
        s.className = 'alt';
        s.textContent = a.clase + ' · ' + (a.confianza * 100).toFixed(1) + '%';
        alts.appendChild(s);
      });
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/clasificar', methods=['POST'])
def clasificar():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400

    f = request.files['imagen']
    if not f.filename:
        return jsonify({'error': 'Archivo vacío'}), 400

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    ruta = tmp.name
    tmp.close()
    f.save(ruta)

    try:
        img = image.load_img(ruta, target_size=(224, 224))
        x = np.expand_dims(preprocess_input(image.img_to_array(img)), axis=0)
        pred = model.predict(x, verbose=0)[0]
    finally:
        os.remove(ruta)

    orden = np.argsort(pred)[::-1]
    idx = int(orden[0])
    conf = float(pred[idx])

    alternativas = [
        {'clase': CLASSES[int(i)], 'confianza': float(pred[int(i)])}
        for i in orden[1:4]
    ]

    if conf < 0.3:
        return jsonify({
            'clase': 'No estoy seguro',
            'confianza': conf,
            'info': {
                'emoji': '🤔', 'color': '#ffb300',
                'donde': 'Vuelve a intentarlo con mejor luz',
                'tips': [
                    'Toma la foto con buena iluminación y fondo neutro.',
                    'Acerca el objeto para que ocupe la mayor parte del encuadre.',
                    'Evita sombras fuertes y objetos de fondo.'
                ],
                'impacto': 'Una imagen clara mejora mucho la precisión del modelo.'
            },
            'alternativas': alternativas
        })

    return jsonify({
        'clase': CLASSES[idx],
        'confianza': conf,
        'info': info_de(CLASSES[idx]),
        'alternativas': alternativas
    })

if __name__ == '__main__':
    app.run(debug=True)
