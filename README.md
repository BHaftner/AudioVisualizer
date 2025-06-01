# 🎶 AudioVisualizer 🎵

### Bring sound to Life

Watch your speech or noise go from your microphone to your screen using visual libraries and fast Fourier transforms.

---

## ✨ Features

* **Real-time Audio Processing:** Watches your audio input and reacts instantly, using Fourier transforms.
* **Bar or Circular Visualizations:** Your audio sounds visualized as bars.

---

### Prerequisites

Before you begin, make sure you have these libraries:

* **SFML:** The Simple and Fast Multimedia Library, version 2.6.x
* **PortAudio:** To capture your audio.
* **Fastest Fourier Transform in the West**

I installed PortAudio, FFTW, and SFML using MSYS2 commands below,
**PortAudio** pacman -S mingw-w64-x86_64-portaudio
**FFTW** pacman -S mingw-w64-x86_64-fftw
**SFML** pacman -S mingw-w64-x86_64-sfml

The imgui and imgui-sfml libraries should install correctly as github submodules.

### Cloning & Building

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/BHaftner/AudioVisualizer.git
    cd AudioVisualizer
    ```

2.  **Compile**
    For compilation feel free to use my tasks.json in the .vscode file if using vs. Otherwise here is the terminal command,
    ```bash
    g++ -g main.cpp imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_widgets.cpp imgui/imgui_tables.cpp imgui-sfml/imgui-SFML.cpp -o AudioVisualizer.exe -Iimgui -Iimgui-sfml -LC:/msys64/mingw64/lib -lsfml-graphics -lsfml-window -lsfml-audio -lsfml-system -lopengl32 -lGdi32 -lws2_32 -lportaudio -lfftw3
    ```
---

## 🎮 How to Use

Once built, simply run the executable! Your default computer microphone will be used.

```bash
./AudioVisualizer # On Linux/macOS
AudioVisualizer.exe # On Windows
```
---
## Pictures
![image](https://github.com/user-attachments/assets/366f177f-eee8-4488-83fc-72a7b596af57)
![image](https://github.com/user-attachments/assets/d3eb922e-65b0-4c45-bd7d-d866f97151bc)


