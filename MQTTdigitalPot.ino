#include <M5StickCPlus2.h>  // Correct library header for M5StickC Plus2  // Include M5StickCPlus2 library
#include <SPI.h>
#include <WiFi.h>
#include <PubSubClient.h>

#define CS_PIN 32        // Chip Select (CS) for MCP4231
#define SCK_PIN 33        
#define MISO_PIN 25     
#define MOSI_PIN 26        

#define SPI_SPEED 1000000  // 1MHz SPI speed
byte addressPot = B00000000;

const char* ssid = "IOT-pecan";
const char* password = "aaaaaaaa";
const char* mqtt_server = "192.168.1.110";
const char* topic = "jc/status/light:0";

WiFiClient espClient;
PubSubClient client(espClient);

void setPot(uint8_t value) {
    digitalWrite(CS_PIN, LOW);  // Select MCP4231
    SPI.transfer(addressPot);         // Command: Write to Wiper 0
    SPI.transfer(value);        // Set wiper position (0-255)
    digitalWrite(CS_PIN, HIGH); // Deselect MCP4231
}

void callback(char* topic, byte* payload, unsigned int length) {
  char payloadStr[10];
  if (length >= sizeof(payloadStr)) length = sizeof(payloadStr) - 1;
  strncpy(payloadStr, (char*)payload, length);
  payloadStr[length] = '\0';

  int potValue = atoi(payloadStr);

  setPot(potValue);
  Serial.print("Wiper set to: ");
  Serial.println(potValue);

  // Display on the M5StickC Plus2 screen
  M5.Lcd.fillScreen(TFT_BLACK);
  M5.Lcd.setCursor(0, 20);
  M5.Lcd.println("Output:");
  M5.Lcd.print(potValue);

}

void setup() {
  M5.begin();  // Initialize M5StickC Plus2
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);  // SCK, MISO (not used), MOSI, CS
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  Serial.begin(115200);
  Serial.println("MCP4231 Digital Potentiometer Test");

  M5.Lcd.setRotation(0);
  M5.Lcd.fillScreen(TFT_BLACK);
  M5.Lcd.setTextColor(TFT_WHITE);
  M5.Lcd.setTextSize(2.5);

  // Set potentiometer to mid-range (127)
  setPot(60);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 10000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to WiFi");
  } else {
    Serial.println("\nWiFi Connection Failed!");
    return;
  }

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("M5StickCPlus2")) {
      Serial.println("Connected!");
      client.subscribe(topic);
    } else {
      Serial.print("Failed, rc=");
      Serial.println(client.state());
      delay(2000);
    }
  }
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Reconnecting to MQTT...");
        if (client.connect("M5StickCPlus2")) {
            Serial.println("Reconnected!");
            client.subscribe(topic);
        } else {
            Serial.print("Failed, rc=");
            Serial.println(client.state());
            delay(5000); // Wait before retrying
        }
    }
}

void loop() {
    if (!client.connected()) {
        reconnect();  // Reconnect to MQTT if disconnected
    }
    client.loop();
}












