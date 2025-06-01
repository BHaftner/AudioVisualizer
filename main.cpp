#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <mutex>

// Graphics and Windowing
#include <SFML/Graphics.hpp>
#include <SFML/System/Clock.hpp>

// Audio Input
#include <portaudio.h>

// Fast Fourier Transform
#include <fftw3.h>

// User Interface
#include "imgui.h"
#include "imgui-sfml.h"

// --- Constants ---

namespace Constants {
    // PortAudio Settings
    constexpr int SAMPLE_RATE = 44100;
    constexpr int FRAMES_PER_BUFFER = 2048;
    constexpr int NUM_CHANNELS = 1;

    // FFTW Settings
    constexpr int FFT_SIZE = FRAMES_PER_BUFFER;

    // PI (Original double precision)
    constexpr double M_PI_D = 3.14159265358979323846;
    // Float precision PI for graphics math
    constexpr float M_PI_F = 3.1415926535f;

    // Visualization Settings
    constexpr int NUM_BARS = 60; // How many bars to display
    constexpr float MIN_FREQ = 20.0f;
    constexpr float MAX_FREQ = 20000.0f;
}

// --- Helper Functions ---

/**
 * @brief Converts HSV (Hue, Saturation, Value) to an SFML RGB Color.
 * @param h Hue, from 0.0 to 1.0.
 * @param s Saturation, from 0.0 to 1.0.
 * @param v Value, from 0.0 to 1.0.
 * @return The corresponding sf::Color.
 */
sf::Color hsvToRgb(float h, float s, float v) {
    h = std::fmod(h, 1.0f);
    if (h < 0.0f) h += 1.0f;
    s = std::clamp(s, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);

    int i = static_cast<int>(h * 6);
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    float r, g, b;

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        default: r = v, g = p, b = q; break;
    }

    return sf::Color(
        static_cast<sf::Uint8>(r * 255),
        static_cast<sf::Uint8>(g * 255),
        static_cast<sf::Uint8>(b * 255)
    );
}

// --- Audio Data Management ---

/**
 * @brief Manages audio samples shared between PortAudio callback and main thread.
 * Uses a mutex for thread-safe access.
 */
class AudioDataBuffer {
public:
    AudioDataBuffer() : samples_(Constants::FRAMES_PER_BUFFER) {}

    void setSamples(const float* inputBuffer, unsigned long frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames > samples_.size()) {
            samples_.resize(frames);
        }
        std::copy(inputBuffer, inputBuffer + frames, samples_.begin());
    }

    // Transfers samples for processing, clearing the internal buffer after copy.
    std::vector<double> getAndClearSamples() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<double> temp_samples(samples_.size());
        std::transform(samples_.begin(), samples_.end(), temp_samples.begin(),
                       [](float s) { return static_cast<double>(s); });
        samples_.clear(); // Clear the source buffer after transfer
        return temp_samples;
    }

private:
    std::vector<float> samples_;
    std::mutex mutex_;
};

// --- PortAudio Callback ---
static AudioDataBuffer s_audioDataBuffer;

static int audioCallback(const void *inputBuffer, void *outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void *userData) {
    if (inputBuffer == nullptr) {
        s_audioDataBuffer.setSamples(nullptr, 0);
        return paContinue;
    }
    s_audioDataBuffer.setSamples(static_cast<const float*>(inputBuffer), framesPerBuffer);
    return paContinue;
}

// --- Main Application ---

enum class VisualizerMode {
    Bars,
    Circle
};

int main() {
    // --- Initialize PortAudio ---
    PaStream *stream = nullptr;
    PaError pa_err = paNoError;

    pa_err = Pa_Initialize();
    if (pa_err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(pa_err) << std::endl;
        return 1;
    }

    PaStreamParameters inputParameters;
    inputParameters.device = Pa_GetDefaultInputDevice();
    if (inputParameters.device == paNoDevice) {
        std::cerr << "Error: No default input device." << std::endl;
        Pa_Terminate();
        return 1;
    }
    inputParameters.channelCount = Constants::NUM_CHANNELS;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    pa_err = Pa_OpenStream(&stream, &inputParameters, nullptr, Constants::SAMPLE_RATE,
                           Constants::FRAMES_PER_BUFFER, paClipOff, audioCallback, nullptr);
    if (pa_err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(pa_err) << std::endl;
        Pa_Terminate();
        return 1;
    }
    
    // --- Initialize FFTW ---
    std::vector<double> fft_in(Constants::FFT_SIZE);
    std::vector<fftw_complex> fft_out(Constants::FFT_SIZE / 2 + 1);
    
    // Custom deleter for fftw_plan
    auto fftw_plan_deleter = [](fftw_plan p) { fftw_destroy_plan(p); };
    std::unique_ptr<fftw_plan_s, decltype(fftw_plan_deleter)> plan(
        fftw_plan_dft_r2c_1d(Constants::FFT_SIZE, fft_in.data(), reinterpret_cast<fftw_complex*>(fft_out.data()), FFTW_ESTIMATE),
        fftw_plan_deleter
    );

    // --- Initialize SFML and ImGui ---
    sf::RenderWindow window(sf::VideoMode(1280, 720), "Enhanced Audio Visualizer");
    window.setFramerateLimit(60);
    
    bool imgui_init_success = ImGui::SFML::Init(window);
    if (!imgui_init_success) {
        std::cerr << "ERROR: ImGui-SFML initialization failed! UI will not be available." << std::endl;
    }
    sf::Clock deltaClock;

    // --- Visualization and UI State Variables ---
    std::vector<float> currentBarHeights(Constants::NUM_BARS, 0.0f);
    float sensitivity = 100.0f;
    float smoothingFactor = 0.18f;
    bool showUi = true;

    VisualizerMode currentMode = VisualizerMode::Bars;
    float circleBaseRadiusFactor = 0.64f;
    float circleBarThickness = 12.0f;


    // --- Start Audio Stream ---
    pa_err = Pa_StartStream(stream);
    if (pa_err != paNoError) {
        std::cerr << "PortAudio error during Pa_StartStream: " << Pa_GetErrorText(pa_err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        if (imgui_init_success) ImGui::SFML::Shutdown();
        return 1;
    }
    std::cout << "PortAudio stream started..." << std::endl;

    // --- Main Loop ---
    while (window.isOpen()) {
        // --- Event Handling ---
        sf::Event event;
        while (window.pollEvent(event)) {
            if (imgui_init_success) ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, static_cast<float>(event.size.width), static_cast<float>(event.size.height));
                window.setView(sf::View(visibleArea));
            } else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::H) {
                showUi = !showUi;
            }
        }

        // --- ImGui UI Definition ---
        if (imgui_init_success) {
            ImGui::SFML::Update(window, deltaClock.restart());

            if (showUi) {
                ImGui::Begin("Visualizer Controls");
                ImGui::Text("Press 'H' to hide this panel");
                ImGui::SliderFloat("Sensitivity", &sensitivity, 10.0f, 300.0f);
                ImGui::SliderFloat("Smoothing", &smoothingFactor, 0.01f, 0.5f);

                ImGui::Separator();
                ImGui::Text("Visualizer Mode:");
                if (ImGui::RadioButton("Bars", currentMode == VisualizerMode::Bars)) {
                    currentMode = VisualizerMode::Bars;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Circle", currentMode == VisualizerMode::Circle)) {
                    currentMode = VisualizerMode::Circle;
                }

                if (currentMode == VisualizerMode::Circle) {
                    ImGui::SliderFloat("Circle Radius Factor", &circleBaseRadiusFactor, 0.05f, 0.5f);
                    ImGui::SliderFloat("Circle Bar Thickness", &circleBarThickness, 1.0f, 10.0f);
                }
                ImGui::Separator();

                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
                ImGui::End();
            }
        }
        
        // --- FFT Processing ---
        std::vector<double> currentAudioSamples = s_audioDataBuffer.getAndClearSamples();

        if (!currentAudioSamples.empty()) {
            // Apply Hann window and copy to FFT input buffer
            size_t samplesToProcess = std::min(currentAudioSamples.size(), static_cast<size_t>(Constants::FFT_SIZE));
            for (size_t i = 0; i < Constants::FFT_SIZE; ++i) {
                if (i < samplesToProcess) {
                    double multiplier = 0.5 * (1 - std::cos(2 * Constants::M_PI_D * i / (Constants::FFT_SIZE - 1))); // Hann window
                    fft_in[i] = currentAudioSamples[i] * multiplier;
                } else {
                    fft_in[i] = 0.0; // Zero-pad if not enough samples
                }
            }
            fftw_execute(plan.get());
        }

        // --- Calculate Bar Heights ---
        const double min_log_freq = std::log(Constants::MIN_FREQ);
        const double max_log_freq = std::log(std::min(static_cast<double>(Constants::MAX_FREQ), static_cast<double>(Constants::SAMPLE_RATE) / 2.0));
        const double log_range = max_log_freq - min_log_freq;

        for (int i = 0; i < Constants::NUM_BARS; ++i) {
            double log_freq_map_factor = (Constants::NUM_BARS == 1) ? 0.5 : (static_cast<double>(i) / (Constants::NUM_BARS - 1));
            double freq_at_bar = std::exp(min_log_freq + log_freq_map_factor * log_range);
            int fft_index = static_cast<int>(freq_at_bar * Constants::FFT_SIZE / Constants::SAMPLE_RATE);

            float targetHeight = 0.0f;
            if (fft_index >= 0 && fft_index < fft_out.size()) {
                double magnitude = std::sqrt(fft_out[fft_index][0] * fft_out[fft_index][0] + fft_out[fft_index][1] * fft_out[fft_index][1]);
                targetHeight = static_cast<float>(magnitude) * sensitivity;
                targetHeight = std::min(targetHeight, static_cast<float>(window.getSize().y)); // Cap height
            }
            currentBarHeights[i] += (targetHeight - currentBarHeights[i]) * smoothingFactor;
            currentBarHeights[i] = std::max(0.0f, currentBarHeights[i]); // Ensure non-negative
        }

        // --- Drawing Logic ---
        window.clear(sf::Color(15, 15, 25)); 

        if (currentMode == VisualizerMode::Bars) {
            float barWidth = static_cast<float>(window.getSize().x) / Constants::NUM_BARS;
            float centerY = static_cast<float>(window.getSize().y) / 2.0f;

            for (int i = 0; i < Constants::NUM_BARS; ++i) {
                float halfHeight = currentBarHeights[i] / 2.0f;
                halfHeight = std::min(halfHeight, centerY); // Prevent bars from exceeding centerY from top/bottom

                sf::RectangleShape upperBar(sf::Vector2f(std::max(1.0f, barWidth - 2.0f), halfHeight));
                upperBar.setPosition(i * barWidth, centerY - halfHeight); 

                sf::RectangleShape lowerBar(sf::Vector2f(std::max(1.0f, barWidth - 2.0f), halfHeight));
                lowerBar.setPosition(i * barWidth, centerY);

                float hue = static_cast<float>(i) / Constants::NUM_BARS;
                float brightness = 0.6f + (currentBarHeights[i] / static_cast<float>(window.getSize().y)) * 0.4f;
                sf::Color barColor = hsvToRgb(hue, 1.0f, brightness);
                
                upperBar.setFillColor(barColor);
                lowerBar.setFillColor(barColor);
                
                window.draw(upperBar);
                window.draw(lowerBar);
            }
        } else if (currentMode == VisualizerMode::Circle) {
            sf::Vector2f center(static_cast<float>(window.getSize().x) / 2.0f, static_cast<float>(window.getSize().y) / 2.0f);
            float min_half_dim = std::min(center.x, center.y);
            float baseRadius = min_half_dim * circleBaseRadiusFactor;

            for (int i = 0; i < Constants::NUM_BARS; ++i) {
                float overallBarLength = currentBarHeights[i];
                float actualHalfLength = overallBarLength / 2.0f;
                
                float angle_rad = (static_cast<float>(i) / Constants::NUM_BARS) * 2.0f * Constants::M_PI_F - (Constants::M_PI_F / 2.0f); // Start from top

                float cosA = std::cos(angle_rad);
                float sinA = std::sin(angle_rad);

                float hue = static_cast<float>(i) / Constants::NUM_BARS;
                float brightnessValue = overallBarLength / static_cast<float>(window.getSize().y);
                float brightness = 0.6f + brightnessValue * 0.4f;
                sf::Color barColor = hsvToRgb(hue, 1.0f, brightness);
                
                float maxInnerExtent = baseRadius - circleBarThickness / 2.0f;
                float maxOuterExtent = min_half_dim - baseRadius - circleBarThickness / 2.0f;
                
                float innerLength = std::clamp(actualHalfLength, 0.0f, maxInnerExtent);
                float outerLength = std::clamp(actualHalfLength, 0.0f, maxOuterExtent);

                // Outer Bar
                if (outerLength > 0.1f) {
                    sf::RectangleShape outerBar(sf::Vector2f(outerLength, circleBarThickness));
                    outerBar.setOrigin(0, circleBarThickness / 2.0f); // Origin at the base touching the conceptual circle
                    outerBar.setPosition(center.x + baseRadius * cosA, center.y + baseRadius * sinA);
                    outerBar.setRotation(angle_rad * 180.0f / Constants::M_PI_F);
                    outerBar.setFillColor(barColor);
                    window.draw(outerBar);
                }

                // Inner Bar
                if (innerLength > 0.1f) {
                    sf::RectangleShape innerBar(sf::Vector2f(innerLength, circleBarThickness));
                    innerBar.setOrigin(innerLength, circleBarThickness / 2.0f); // Origin at the tip that extends inwards
                    innerBar.setPosition(center.x + baseRadius * cosA, center.y + baseRadius * sinA);
                    innerBar.setRotation(angle_rad * 180.0f / Constants::M_PI_F);
                    innerBar.setFillColor(barColor);
                    window.draw(innerBar);
                }
            }
        }
        
        if (imgui_init_success) ImGui::SFML::Render(window);
        window.display();
    }

    // --- Cleanup ---

    std::cout << "Stopping PortAudio stream..." << std::endl;
    if (stream) {
        pa_err = Pa_StopStream(stream);
        if (pa_err != paNoError) std::cerr << "PortAudio error during Pa_StopStream: " << Pa_GetErrorText(pa_err) << std::endl;
        
        pa_err = Pa_CloseStream(stream);
        if (pa_err != paNoError) std::cerr << "PortAudio error during Pa_CloseStream: " << Pa_GetErrorText(pa_err) << std::endl;
    }
    pa_err = Pa_Terminate();
    if (pa_err != paNoError) std::cerr << "PortAudio error during Pa_Terminate: " << Pa_GetErrorText(pa_err) << std::endl;
    
    if (imgui_init_success) ImGui::SFML::Shutdown();
    std::cout << "Cleanup complete. Exiting." << std::endl;

    return 0;
}