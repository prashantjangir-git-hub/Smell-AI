/*
  AI that can smell - BME688 reader for ESP32

  Wiring (I2C):
    BME688 VIN/VCC -> ESP32 3V3
    BME688 GND     -> ESP32 GND
    BME688 SCL     -> ESP32 GPIO 22
    BME688 SDA     -> ESP32 GPIO 21

  Libraries (Arduino IDE -> Tools -> Manage Libraries):
    "Adafruit BME680 Library" (it also works for the BME688)
    "Adafruit Unified Sensor"

  Board: "ESP32 Dev Module" (install "esp32 by Espressif" in Boards Manager).

  Every cycle the heater steps through a few temperatures. Each temperature
  makes the sensor react a bit differently, so one chip acts like several
  sensors. One line is printed per cycle:

    BME,<millis>,<temp C>,<humidity %>,<R at 200C>,<R at 250C>,<R at 300C>,<R at 350C>

  record_bme688.py on your laptop reads these lines.
*/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"

const int HEATER_STEPS[] = {200, 250, 300, 350};   // degrees C
const int N_STEPS = sizeof(HEATER_STEPS) / sizeof(HEATER_STEPS[0]);
const int HEATER_MS = 150;                          // how long to heat at each step

Adafruit_BME680 bme;

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(21, 22);  // SDA, SCL

  // Most breakouts use address 0x77, some use 0x76
  if (!bme.begin(0x77) && !bme.begin(0x76)) {
    Serial.println("# ERROR: BME688 not found. Check the wiring.");
    while (true) delay(1000);
  }
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);

  Serial.print("# READY steps=");
  for (int i = 0; i < N_STEPS; i++) {
    Serial.print(HEATER_STEPS[i]);
    if (i < N_STEPS - 1) Serial.print(";");
  }
  Serial.println();
}

void loop() {
  float temperature = NAN, humidity = NAN;
  uint32_t resistance[N_STEPS];

  for (int i = 0; i < N_STEPS; i++) {
    bme.setGasHeater(HEATER_STEPS[i], HEATER_MS);
    if (!bme.performReading()) {
      Serial.println("# WARN: reading failed");
      return;
    }
    resistance[i] = bme.gas_resistance;   // ohms
    temperature = bme.temperature;
    humidity = bme.humidity;
  }

  Serial.print("BME,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(temperature, 2);
  Serial.print(",");
  Serial.print(humidity, 2);
  for (int i = 0; i < N_STEPS; i++) {
    Serial.print(",");
    Serial.print(resistance[i]);
  }
  Serial.println();
}
