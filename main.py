import subprocess
import recognize

def main():
    print("[🚗] Шаг 1: Обнаружение номеров...")
    subprocess.run(["python", "detect.py"])

    print("[🔤] Шаг 2: Распознавание текста...")
    recognize.recognize_text()

if __name__ == "__main__":
    main()
