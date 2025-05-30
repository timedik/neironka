import os
import pytesseract
import json
import cv2

def recognize_text(input_dir='outputs'):
    # Проверяем существование папки и создаём, если её нет
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"[!] Папка '{input_dir}' не найдена. Создана новая папка.")

    results = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            text = pytesseract.image_to_string(img, config='--psm 7')
            text = text.strip()

            results.append({
                "filename": filename,
                "text": text
            })
            print(f"[+] {filename}: {text}")

    # Сохраняем в JSON
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("[✓] Распознавание завершено. Результаты в results.json")

if __name__ == "__main__":
    recognize_text()
