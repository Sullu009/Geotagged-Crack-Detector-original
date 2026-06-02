import os
import time
import logging
import cv2
import numpy as np
import requests
from gps_helper import get_gps
from tensorflow.lite.python.interpreter import Interpreter

# ---------- Telegram config ----------
TELEGRAM_BOT_TOKEN = "8594741286:AAExb-22mD0KEn_mZiwMOyeALoBhWAd77i8"
TELEGRAM_CHAT_ID = 5257066612  # your chat id (integer)

ALERT_COOLDOWN = 15  # seconds between alerts
last_alert_time = 0


def send_telegram_alert(image_path, label, pred_score, gps_text):
    """Send a crack alert photo with caption via Telegram."""
    global last_alert_time

    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        logging.info("Skipping alert due to cooldown")
        return

    last_alert_time = now

    caption_lines = [
        f"Alert: {label}",
        f"Confidence: {pred_score * 100:.1f}%",
    ]
    if gps_text:
        caption_lines.append(gps_text)

    caption = "\n".join(caption_lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            resp = requests.post(url, files=files, data=data, timeout=10)
            logging.info(f"Telegram response: {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")


def load_interpreter():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "model", "quantized-model.lite")

    interpreter = Interpreter(model_path=model_file, num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization params
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_0_scale, output_0_zero_point = output_details[0]["quantization"]
    output_1_scale, output_1_zero_point = output_details[1]["quantization"]

    return (
        interpreter,
        input_details,
        output_details,
        input_scale,
        input_zero_point,
        output_0_scale,
        output_0_zero_point,
        output_1_scale,
        output_1_zero_point,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Loading TFLite model...")
    (
        interpreter,
        input_details,
        output_details,
        input_scale,
        input_zero_point,
        output_0_scale,
        output_0_zero_point,
        output_1_scale,
        output_1_zero_point,
    ) = load_interpreter()

    # Determine input size
    _, in_h, in_w, _ = input_details[0]["shape"]

    logging.info("Opening camera /dev/video0 ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    alert_image_path = os.path.join(script_dir, "crack_alert_headless.jpg")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Rotate if needed like GUI version
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Prepare input
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (in_w, in_h))
            img_norm = img_resized / 255.0
            img_norm = img_norm.astype(np.float32)

            # Quantize
            img_q = (img_norm / input_scale) + input_zero_point
            input_data = np.expand_dims(img_q, axis=0).astype(input_details[0]["dtype"])

            # Inference
            start_t = time.time()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            # Outputs
            output_0_tensor = interpreter.tensor(output_details[0]["index"])
            output_1_tensor = interpreter.tensor(output_details[1]["index"])

            output_1 = output_1_scale * (
                output_1_tensor().astype(np.float32) - output_1_zero_point
            )

            pred_class = int(np.argmax(np.squeeze(output_1)))
            pred_score = float(np.squeeze(output_1)[pred_class])

            infer_ms = (time.time() - start_t) * 1000.0
            logging.info(
                f"Inference: class={pred_class} score={pred_score*100:.1f}% time={infer_ms:.1f}ms"
            )

            label = "Unknown"
            if pred_class == 1:
                label = "Crack"
            elif pred_class == 0:
                label = "No Crack"

            gps_text = ""
            # Only alert for crack with decent confidence
            if pred_class == 1 and pred_score > 0.7:
                # Get GPS location
                lat, lon = get_gps(timeout=8)
                if lat is not None and lon is not None:
                    gps_text = f"GPS: {lat:.6f}, {lon:.6f}"
                else:
                    gps_text = "GPS: No Fix"

                # Draw some text on the frame for context
                vis = frame.copy()
                cv2.putText(
                    vis,
                    label,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"{pred_score*100:.1f}%",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if gps_text:
                    cv2.putText(
                        vis,
                        gps_text,
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Save and send
                cv2.imwrite(alert_image_path, vis)
                logging.info("Crack detected, sending Telegram alert...")
                send_telegram_alert(alert_image_path, label, pred_score, gps_text)

            # Small sleep to avoid max CPU
            time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
