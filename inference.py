import datetime
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def hospital_inference(model, img_path, threshold, model_version="1.0"):
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(224,224)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = densenet_preprocess(img)

    prob = float(model.predict(img)[0][0])
    pred_label = 1 if prob >= threshold else 0
    pred_text = 'PNEUMONIA' if pred_label == 1 else 'NORMAL'

    return {
        "exam_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_version": model_version,
        "prediction_label": pred_label,
        "prediction_text": pred_text,
        "probability_pneumonia": prob,
        "clinical_threshold": threshold,
        "recommendation": (
            "Encaminhar para avaliação médica imediata"
            if pred_label == 1 else
            "Manter acompanhamento clínico"
        )
    }
