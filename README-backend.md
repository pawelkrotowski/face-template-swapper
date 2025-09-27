
# Dokumentacja backendu – Face Template InsightFace

Ten dokument opisuje **backend** projektu obsługujący wykrywanie twarzy, wyrównywanie/łączenie oraz **zamianę twarzy** z użyciem modeli InsightFace (w tym InSwapper). Dokument dotyczy komponentu `inference-face` oraz konfiguracji bramki Nginx (`gateway`).

> ✅ **Skrót**: wystawiamy API przez Nginx na `http://<HOST>:8080`. Ścieżki zewnętrzne mają prefiks **`/infer/`**. Wewnątrz kontenera aplikacja FastAPI definiuje ścieżki także z prefiksem `/infer/*`, a Nginx zdejmuje pierwszy prefiks – dlatego z zewnątrz wołamy **`/infer/infer/*`**.

---

## Spis treści

1. [Architektura i komponenty](#architektura-i-komponenty)
2. [Szybki start](#szybki-start)
3. [Środowisko i wymagania](#środowisko-i-wymagania)
4. [Modele i cache](#modele-i-cache)
5. [Endpointy API](#endpointy-api)
6. [Przykłady użycia (cURL)](#przykłady-użycia-curl)
7. [Konfiguracja Docker / Compose](#konfiguracja-docker--compose)
8. [GPU vs CPU](#gpu-vs-cpu)
9. [Rozwiązywanie problemów](#rozwiązywanie-problemów)
10. [Wydajność i jakość](#wydajność-i-jakość)
11. [Bezpieczeństwo i uwagi prawne](#bezpieczeństwo-i-uwagi-prawne)
12. [Licencje i podziękowania](#licencje-i-podziękowania)

---

## Architektura i komponenty

- **gateway** – Nginx wystawiający port 80 na hoście (mapowany na `8080` wg Compose). Proxy do:
  - `/` → **web** (aplikacja React – opcjonalnie),
  - `/api/` → **api** (skafold HTTP, jeżeli używany),
  - `/infer/` → **inference-face** (FastAPI z InsightFace).
- **inference-face** – usługa FastAPI:
  - wykrywanie twarzy (SCRFD z pakietu `buffalo_l`),
  - landmarki 5/106 punktów,
  - proste wyrównywanie i blendowanie (endpointy `align`, `align_v2`),
  - **InSwapper (`inswapper_128.onnx`)** do wysokiej jakości „face swap”,
  - obsługa **ONNX Runtime** z auto-wykryciem dostawców (`CUDAExecutionProvider` / `CPUExecutionProvider`).

Struktura katalogów (wycinek):
```
face-template-insightface/
├─ docker-compose.yml
├─ infra/
│  └─ nginx.conf                # konfiguracja gateway
├─ services/
│  ├─ inference-face/
│  │  ├─ Dockerfile
│  │  ├─ requirements.txt
│  │  └─ main.py                # FastAPI + InsightFace + InSwapper
│  └─ web/ …                    # (opcjonalnie) aplikacja React
├─ data/
│  ├─ uploads/                  # wrzutki (podgląd/diagnostyka)
│  └─ outputs/                  # wyniki PNG
└─ models_cache/                # mapowane do /root/.insightface
   └─ models/
      ├─ buffalo_l/…
      └─ inswapper_128.onnx     # model InSwapper (ONNX)
```

---

## Szybki start

```bash
# 1) Zbuduj i uruchom
docker compose up -d --build

# 2) Zdrowie endpointów (z hosta)
curl -s http://localhost:8080/infer/infer/health | jq .

# 3) Test wykrycia landmarków
curl -s -F image=@face.jpg http://localhost:8080/infer/infer/landmarks | jq .

# 4) Test zamiany twarzy (InSwapper)
curl -s -F template=@template.jpg -F portrait=@face.jpg \
  -F download=1 \
  http://localhost:8080/infer/infer/swap_inswapper -o swapped.png && file swapped.png
```

> **Uwaga**: pierwsze wywołanie pobierze modele do `models_cache/` (jeśli są dostępne online) albo użyje już wgranych plików.

---

## Środowisko i wymagania

- **Docker + Docker Compose**
- **Python** w obrazie (wewnątrz kontenera – instalowany przez Dockerfile)
- **ONNX Runtime** (`onnxruntime-gpu` lub `onnxruntime`)
- Systemowe biblioteki dla OpenCV (`libgl1`, `libglib2.0-0`)
- Dla GPU: sterownik NVIDIA + `nvidia-container-toolkit` na hoście

---

## Modele i cache

- Domowy katalog modeli InsightFace w kontenerze: **`/root/.insightface`** (mapowany z hosta `./models_cache`).
- **Pakiet buffalo_l** (SCRFD + landmarki + rozpoznawanie) – pobiera się automatycznie przy pierwszym użyciu.
- **InSwapper**: plik **`models_cache/models/inswapper_128.onnx`** (rekomendowany).  
  - Jeśli auto-pobranie nie działa, wgraj ręcznie (sprawdź rozmiar, nie może to być plik HTML/tekst).

---

## Endpointy API

> Z **zewnątrz** (przez Nginx) wywołujemy ścieżki `http://<HOST>:8080/infer/infer/...`

### `GET /infer/infer/health`
Informacja diagnostyczna.
**Odpowiedź (JSON):**
```json
{
  "ok": true,
  "service": "inference-face",
  "onnx_providers_available": ["CUDAExecutionProvider","CPUExecutionProvider"],
  "using_providers": ["CUDAExecutionProvider","CPUExecutionProvider"],
  "ctx_id": 0,
  "insightface_home": "/root/.insightface"
}
```

### `POST /infer/infer/landmarks`
Wykrycie twarzy i landmarków.

**Form-Data**: `image` (plik JPG/PNG)  
**Odpowiedź (JSON):**
```json
{
  "faces": [
    {
      "det_score": 0.99,
      "bbox": [x1,y1,x2,y2],
      "landmarks_106": [[x,y], ...]
    }
  ],
  "best_index": 0
}
```

### `POST /infer/infer/align` (szybkie wyrównanie 5-punktowe)
Proste dopasowanie twarzy z `portrait` do twarzy w `template` + blend.

**Form-Data**:  
- `template` (plik),  
- `portrait` (plik),  
- `blend` (`seamless`|`alpha`, domyślnie `seamless`),  
- `download` (`0`|`1` – czy zwrócić PNG bezpośrednio).

**Odpowiedź**: PNG (gdy `download=1`) lub
```json
{ "ok": true, "output_path": "/data/outputs/swap_*.png" }
```

### `POST /infer/infer/align_v2` (lepsze maski + 106 pkt + korekcja koloru)
Jak wyżej, ale: dopasowanie po 106 punktach, maska z convex hull, korekcja barw (Lab). Parametry takie same.

### `POST /infer/infer/swap_inswapper` (**zalecane**)
Wysokiej jakości „face swap” z użyciem **InSwapper** (ONNX).

**Form-Data**:  
- `template` (plik – obraz docelowy),  
- `portrait` (plik – obraz źródłowy, z którego bierzemy tożsamość),  
- `download` (`0`|`1`).

**Odpowiedź**: PNG (gdy `download=1`) lub
```json
{ "ok": true, "output_path": "/data/outputs/swap_inswapper_*.png" }
```

**Uwagi**:  
- Przy wielu twarzach w obrazie wybierana jest ta z najwyższym `det_score`. Można rozszerzyć endpoint o `source_index`/`target_index`.

---

## Przykłady użycia (cURL)

```bash
# Zdrowie
curl -s http://localhost:8080/infer/infer/health | jq .

# Landmarki
curl -s -F image=@face.jpg http://localhost:8080/infer/infer/landmarks | jq .

# Align (PNG strumieniowo)
curl -s -F template=@template.jpg -F portrait=@face.jpg -F blend=alpha -F download=1 \
  http://localhost:8080/infer/infer/align -o out_align.png

# Align_v2 (PNG strumieniowo)
curl -s -F template=@template.jpg -F portrait=@face.jpg -F blend=seamless -F download=1 \
  http://localhost:8080/infer/infer/align_v2 -o out_align_v2.png

# InSwapper (rekomendowane)
curl -s -F template=@template.jpg -F portrait=@face.jpg -F download=1 \
  http://localhost:8080/infer/infer/swap_inswapper -o out_swap.png
```

---

## Konfiguracja Docker / Compose

Najważniejsze fragmenty:
- **`services/inference-face/Dockerfile`** – obraz oparty o CUDA runtime lub Python slim:
  - dla GPU najwygodniej użyć `nvidia/cuda:XX-runtime-ubuntu22.04` i `onnxruntime-gpu`,
  - dla CPU – `python:3.11-slim` i `onnxruntime`.
- **`docker-compose.yml`** – wolumeny i GPU:
  ```yaml
  inference-face:
    build: ./services/inference-face
    volumes:
      - ./data:/data
      - ./models_cache:/root/.insightface
    environment:
      - INSIGHTFACE_HOME=/root/.insightface
    # GPU:
    gpus: all
  ```
- **`infra/nginx.conf`** – routing:
  ```nginx
  location /infer/ { proxy_pass http://inference-face:8001/; }
  ```

---

## GPU vs CPU

- **GPU** (szybciej): wymagane sterowniki NVIDIA i `nvidia-container-toolkit` na hoście, `gpus: all` w Compose oraz `onnxruntime-gpu` w wymaganiach Pythona.
- **CPU** (prościej): usuń `gpus: all` i użyj `onnxruntime` zamiast `onnxruntime-gpu`.

Aplikacja sama wybierze provider:
- `CUDAExecutionProvider` + `CPUExecutionProvider` (GPU dostępne),
- tylko `CPUExecutionProvider` (fallback).

---

## Rozwiązywanie problemów

**502 Bad Gateway (Nginx)**  
- Sprawdź logi modelu: `docker compose logs -f inference-face`
- Najczęściej: kontener `inference-face` nie wstał (błąd przy starcie, brak zależności).

**`python-multipart` wymagany**  
- Dodaj do `requirements.txt` w `inference-face`: `python-multipart==0.0.9`.

**CUDA/ORT: `libcublasLt.so.11` nie znaleziono**  
- Obraz nie ma bibliotek CUDA. Użyj bazy `nvidia/cuda:*runtime*` albo przejdź na CPU (patrz sekcja GPU vs CPU).

**`InvalidProtobuf: Protobuf parsing failed` dla `inswapper_128.onnx`**  
- Plik jest uszkodzony / to nie jest ONNX (często HTML 404 zapisany jako `.onnx`).  
- Usuń i pobierz ponownie. Sprawdź `file inswapper_128.onnx` oraz rozmiar (dziesiąt MB).

**Model InSwapper „not available”**  
- Zrestartuj usługę (ładowanie przy starcie) lub użyj „lazy loadera” w endpointzie.  
- Upewnij się, że plik jest w `models_cache/models/inswapper_128.onnx` (mapuje się na `/root/.insightface/models/…`).

**Ścieżki 404 (`Not Found`)**  
- Pamiętaj o **podwójnym `infer`** z zewnątrz: `/infer/infer/...` (Nginx zdejmuje pierwszy prefiks).  
- Sprawdź dostępne ścieżki: `curl -s http://localhost:8080/infer/infer/health` i `openapi.json`.

---

## Wydajność i jakość

- **det_size**: zwiększ do `(1024, 1024)` w `main.py`, jeśli twarze są małe lub oddalone.
- **align_v2**: używa 106 punktów + convex hull + korekcję koloru – lepsze dopasowanie do szablonów fotograficznych.
- **InSwapper**: najlepsza jakość przy minimalnym strojenie. Dla wielu twarzy w obrazie można dodać parametry wyboru indeksów (`source_index`, `target_index`).

---

## Bezpieczeństwo i uwagi prawne

- Dane obrazowe mogą zawierać informacje biometryczne; zachowaj zgodność z RODO/GDPR i lokalnym prawem.
- Nie przechowuj dłużej niż to konieczne; katalog `data/uploads`/`data/outputs` może być czyszczony okresowo (cron).  
- Ogranicz dostęp do API (IP allowlist, auth reverse proxy) w środowisku publicznym.

---

## Licencje i podziękowania

- **InsightFace** (modele i biblioteka) – sprawdź licencję projektu i ograniczenia korzystania z modeli.
- **OpenCV**, **FastAPI**, **ONNX Runtime**, **Uvicorn**, **Nginx** – odpowiednie licencje open‑source.
- W projekcie zastosowano publicznie dostępne artefakty modeli dla celów POC.

---

## Kontakt / rozwój

- Dodaj zadania: wybór konkretnej twarzy, galeria predefiniowanych szablonów, kolejkowanie zadań, trwałe logowanie metadanych, ograniczanie rozmiaru wejściowych obrazów.
- Pull requests i issue tracker w repozytorium projektu.
