import pyautogui
import time

INTERVAL = 240  # 4 minutes

print("Auto-scroll started. Keep the Kaggle tab focused.")
print("Press Ctrl+C in terminal to stop.")

try:
    while True:
        # small scroll down
        pyautogui.scroll(-300)

        time.sleep(1)

        # small scroll up
        pyautogui.scroll(300)

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("Stopped.")