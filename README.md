# qr_dock_kf — Docking a QR para el Puzzlebot (TCSM)

Paquete de **docking de precisión a un QR fijo**, incluido el caso difícil:
**el QR rotado sobre su eje vertical (Z)**, es decir, cuando el robot no llega
perpendicular a él.

Contiene dos enfoques (uno descartado, uno que funciona) y un simulador
sintético para probarlos sin Gazebo.

> **TL;DR — usa `qr_dock_map_node`** (asistido por localización). El docking
> puramente visual (`qr_dock_kf_node`) resultó poco confiable; ver
> [Historia y por qué](#historia-y-por-qué).

---

## El problema

Estimar la **normal del plano del QR** (su rotación en Z) desde una sola cámara
es frágil: la ambigüedad de doble solución del PnP plano y el ruido de esquinas
hacen que el *yaw* salte. Peor aún, en un acercamiento oblicuo el **QR se sale
del FOV** a media maniobra y la recuperación diverge — esto le pasa por igual a
cualquier docking puramente visual (lo confirmamos con `qr_pose_align` y
`qr_dock_kf` en el simulador: ~50 % de éxito).

## La solución: `qr_dock_map_node` (asistido por localización)

**La maniobra grande va por localización global (EKF/MCL/ArUco), no por
visión. La visión solo afina, de lejos y de frente.** Así el QR puede salir del
FOV durante la maniobra sin que importe.

Validado en simulación: **12/12 dockeos** (tilts 0/±25/±30, arranques
oblicuos/lejanos), **determinista**, y **robusto al error de localización**
(sesgo constante se cancela; drift 1 cm/s y jitter 2 cm → siguen dando
0.5–1.1 cm lateral, porque la precisión final la da la corrección visual, no la
localización). Standoff ±1 mm, heading ~4–5°.

### Máquina de estados

```
OBSERVE ──▶ NAV_MAP ──▶ REFINE ──▶ COMMIT ──▶ DONE
 (fija QR    (llega de    (corrige    (avance
  en map)     frente por   visual de   recto a
              localización) cerca)      ciegas)
```

| Fase | Qué hace | ¿Usa visión? |
|------|----------|:---:|
| **OBSERVE** | Detecta el QR (`solvePnP` + desambiguación `n_cam_z<0`), lo transforma a frame `map` con la TF `map→base_footprint` y **fija su pose** cuando es estable. Calcula el **pre-grasp** (a `pregrasp_dist_m` enfrente del QR sobre su normal). Si no lo ve, gira despacio para buscarlo. | Sí (de lejos: QR chico/centrado) |
| **NAV_MAP** | Conduce al pre-grasp con controlador polar **rho-α-β**, usando la TF `map→base` como realimentación. Termina **perpendicular y centrado**. | **No** (por eso da igual perder el QR) |
| **REFINE** | Ya de frente, re-adquiere el QR y rota para centrarlo lateralmente (`my→0`), absorbiendo el error de localización. Si no reaparece, hace COMMIT confiando en la localización. | Sí (de frente: QR chico/centrado) |
| **COMMIT** | Avanza recto por **odometría** hasta `marker_gap_m` del QR. Open-loop a propósito (el QR ya está demasiado cerca para verse). | No |
| **DONE** | Publica `/align/done=true` y se detiene. | — |

### Distancias (con `pregrasp_dist_m=0.45`, `marker_gap_m=0.15`, cámara 0.14 m adelante)

| Momento | base_link → QR | cámara → QR | ¿Necesita ver el QR? |
|---------|:---:|:---:|:---:|
| OBSERVE | ~0.8–1.1 m | ~0.7–1.0 m | Sí |
| **REFINE (última mirada)** | **0.45 m** | **~0.31 m** | Sí |
| Final (tras COMMIT) | 0.15 m | ~0.01 m | **No (open-loop)** |

> La visión solo se necesita hasta el pre-grasp. Que la cámara "no vea bien el
> QR de cerca" **no importa** — el COMMIT es a ciegas. Sube `pregrasp_dist_m`
> si tu QR cuesta detectarlo de cerca.

---

## Fuente de la pose del QR: interna vs. on-Jetson (`/qr/pose`)

El nodo puede obtener la pose del QR de dos formas (`qr_pose_source`):

- **`internal`** (default): detecta y hace `solvePnP` sobre la imagen aquí mismo.
  Útil en sim o sin la Jetson. Detecta sobre la imagen **downsized**.
- **`external`**: consume **`/qr/pose`** del nodo on-Jetson
  (`optimized_camera_node_undistort_qr` del paquete `qr_detection`), que detecta
  sobre el frame **full-size** → mejor pose, y descarga al PC. El nodo:
  - transforma `/qr/pose` (frame `camera`) a `base`/`map` con **TF**,
  - filtra por `target_qr_id` (payload `{"id": N}`),
  - usa `qr_normal_axis` (`x` por defecto, la convención de `qr_detection`:
    +X sale de la cara del QR) para la normal,
  - enciende/apaga la detección de la Jetson publicando en **`/qr_enable`**
    (`1` al activar, `0` al terminar) si `manage_qr_enable=true`.

  **Requisito:** el frame de `/qr/pose` (`camera`) debe estar en el árbol TF
  (URDF / EKF / la TF estática que publica `qr_debug_viz`). Validado en sim:
  dockea con 0.5–0.9 cm igual que el path interno.

  ```bash
  ros2 launch qr_dock_kf dock_with_localization.launch.py \
      qr_pose_source:=external target_qr_id:=1 qr_normal_axis:=x
  ```

## Interfaz (`qr_dock_map_node`)

| Dirección | Nombre | Tipo |
|-----------|--------|------|
| sub | `image_topic` (`/image_raw` o `/image_raw/compressed`) | `Image` / `CompressedImage` |
| sub | `odom_topic` (`/odometry/filtered`) | `nav_msgs/Odometry` |
| sub | `/align/mode` | `std_msgs/String` (activa con `dock_qr_map`) |
| TF  | `map → base_footprint` | (de tu localización) |
| pub | `alignment_cmd_vel` | `geometry_msgs/Twist` (lo mezcla `cmd_vel_mux`) |
| pub | `/align/done` | `std_msgs/Bool` |
| srv | `/qr_dock_map/enable` | `custom_interfaces/SetProcessBool` |
| pub | `~/debug_image`, `~/qr_marker`, `~/pregrasp` | debug / RViz |

### Parámetros clave

| Parámetro | Default | Qué controla |
|-----------|:---:|--------------|
| `pregrasp_dist_m` | 0.45 | Distancia mínima a la que aún ve el QR (REFINE) |
| `marker_gap_m` | 0.15 | Standoff final (base→QR) |
| `cam_x/y/z_offset_m`, `cam_pitch_deg` | 0.05/0/0.205/0 | **Extrínsecos de la cámara — MEDIR en el robot** |
| `use_compressed_image` | false | Cámara comprimida (`/image_raw/compressed`) |
| `default_qr_size_mm` | 97 | Tamaño del QR (o embebe `qr_mm=NN` en el payload) |
| `calib_path` | "" | `.npz` de calibración (si vacío, usa intrínsecas fallback) |
| `observe_frames`, `observe_*_std` | 12 / 0.03 / 0.08 | Rigor del lock del QR en `map` |
| `map_frame`, `base_frame` | map / base_footprint | Frames TF |

---

## Nodos de los que depende (stack de localización)

`dock_with_localization.launch.py` levanta esto; el docking **consume** sus TFs:

| Nodo (paquete) | Rol | Publica |
|----------------|-----|---------|
| `odometry_node` (movement_control) | integra encoders | `/odom` |
| `ekf_node` (ekf_optimized) | **Kalman**: fusiona odom + ArUco | `/odometry/filtered`, TF `odom→base_footprint` |
| `mcl_v3` (mcl_optimized) | Monte-Carlo con lidar vs mapa | TF `map→odom` |
| `aruco_localizer_node` (aruco_detection_optimized) | ArUcos conocidos → corrige EKF a `map` | `/aruco/robot_pose_odom` |
| `map_publisher_node` (tcsm_sim) | publica el mapa | `/map` |
| `robot_state_publisher` + 3× `static_transform_publisher` | TFs URDF y estáticos | cadena a `camera_link` |
| `cmd_vel_mux` (movement_control) | mezcla velocidades | consume `alignment_cmd_vel` |

Cadena TF resultante: **`map → odom → base_footprint → base_link → camera_link`**.
EKF (Kalman) + MCL + ArUco dicen dónde está el robot en el mapa; el docking usa
eso para llegar de frente al QR y la visión + commit clavan el último tramo.

---

## Cómo correr

### Simulación (sin Gazebo, sintético)

```bash
cd ~/Github/TCSM && source install/setup.bash

# arquitectura por localización (la buena):
ros2 launch qr_dock_kf qr_dock_map_sim.launch.py qr_z_tilt_deg:=30.0
ros2 launch qr_dock_kf qr_dock_map_sim.launch.py robot_x:=-0.4 qr_z_tilt_deg:=20.0

# ver cámara + HUD:
ros2 run rqt_image_view rqt_image_view /qr_dock_map_node/debug_image
```

El simulador (`qr_sim_world`) renderiza un QR virtual con rotación Z
configurable, modela el robot (uniciclo), publica `Image`+`CompressedImage`+
`Odometry` y la TF ground-truth `map→odom→base`. Estresores de localización:
`-p loc_bias_x/y -p loc_drift_x/y -p loc_jitter_m`.

### Robot real (end-to-end)

```bash
ros2 launch qr_dock_kf dock_with_localization.launch.py \
    calib_path:=/home/edrick/Github/TCSM/src/tcsm_camera_utils/data/calibration_data/calibration_data_3_PERFECT.npz \
    use_compressed_image:=true image_topic:=/image_raw/compressed

# activar el docking:
ros2 topic pub -1 /align/mode std_msgs/msg/String "{data: dock_qr_map}"
# o:  ros2 service call /qr_dock_map/enable custom_interfaces/srv/SetProcessBool "{enable: true}"
```

#### Checklist robot real (pendiente de validar)
- [ ] **Medir** los extrínsecos de la cámara (`cam_*_offset_m`) — hay valores inconsistentes en el repo.
- [ ] Pasar tu **calibración real** (`calib_path`).
- [ ] Confirmar el **topic de imagen** (`/image_raw` crudo o `/image_raw/compressed`).
- [ ] La localización debe converger: el **mapa** (`map9.yaml`) y los **11 ArUcos físicos** deben coincidir con `aruco_mapped2.yaml`, y necesita `/scan` (lidar) + encoders.
- [ ] Ajustar `pregrasp_dist_m` al rango de detección de tu QR de 97 mm.

---

## Checklist de verificación de TF (al conectar la Jetson)

El modo `external` necesita que `/qr/pose` (frame `camera`) sea **TF-resoluble**
hasta `base_footprint` y `map`. Verifica en este orden:

```bash
# 1) ¿Llega la pose y el payload de la Jetson? (recuerda /qr_enable=1)
ros2 topic pub /qr_enable std_msgs/msg/Int32 "{data: 1}" --once
ros2 topic echo --once /qr/pose      # ¿hay PoseStamped? anota header.frame_id (ej. "camera")
ros2 topic echo --once /qr/data      # ¿{"id": N}?

# 2) ¿El frame de /qr/pose está en el árbol TF?
ros2 run tf2_tools view_frames       # genera frames.pdf; busca "camera" conectado
ros2 run tf2_ros tf2_echo base_footprint camera   # ¿resuelve sin error?
ros2 run tf2_ros tf2_echo map base_footprint      # ¿la localización publica esto?

# 3) Cadena completa que usa el docking:
ros2 run tf2_ros tf2_echo map camera   # map -> camera debe resolver
```

Reglas:
- Si `tf2_echo base_footprint camera` falla → falta publicar `base_link→camera`.
  Lo publica `qr_debug_viz` (`use_static_tf:=true`) o ponlo en el URDF/EKF.
- Si el `frame_id` de `/qr/pose` no es `camera` (p. ej. `camera_link`), no
  pasa nada: el nodo usa el `frame_id` que venga en el mensaje. Solo asegúrate
  de que **ese** frame esté en el TF.
- Si `tf2_echo map base_footprint` falla → la localización (EKF/MCL/ArUco o
  QR) no está corriendo o no converge. Para una primera prueba usa
  `dock_odom_only.launch.py` (pone `map=odom`, no necesita ArUcos).
- Si dockea pero queda girado ~180° o la normal sale rara → prueba
  `qr_normal_axis:=z`.

Sanity de detección (independiente del TF):
```bash
ros2 launch qr_detection qr_debug_viz.launch.py \
  image_topic:=/image_raw/compressed use_compressed_image:=true
# en RViz: /qr/debug_image (polígono+texto) y /qr/pose_marker (cubo)
```

## Contenido del paquete

| Archivo | Qué es |
|---------|--------|
| `qr_dock_kf/qr_dock_map_node.py` | **Nodo recomendado** (asistido por localización) |
| `qr_dock_kf/qr_dock_kf_node.py` | Docking puramente visual (KF + ego-movimiento). **Descartado** (~50 % éxito) |
| `qr_dock_kf/qr_sim_world.py` | Simulador sintético cámara+QR+robot (sin Gazebo) |
| `launch/qr_dock_map_sim.launch.py` | Sim de la arquitectura por localización |
| `launch/dock_with_localization.launch.py` | **End-to-end robot real** (localización + docking) |
| `launch/qr_dock_kf_sim.launch.py` | Sim del docking puramente visual (histórico) |
| `config/sim_calib.npz` | Intrínsecas compartidas para el sim (80° FOV) |

---

## Historia y por qué

1. **`qr_dock_kf_node`** (visual puro, KF con predicción por ego-movimiento,
   inspirado en `dawan0111/Auto-Marker-Docking`): dockea con precisión cuando
   funciona, pero el éxito es **estocástico (~50 %)** — un frame ruidoso dispara
   un giro que pierde el QR del FOV y el coasting a ciegas diverge.
2. Probamos también el `qr_pose_align` existente en el mismo sim: **falla
   igual**. Conclusión: el cuello de botella es **perder el QR del FOV en el
   acercamiento** (geometría/percepción), no el controlador.
3. **`qr_dock_map_node`** (asistido por localización): resuelve el problema
   moviendo la maniobra grande a la localización global y dejando la visión solo
   para la corrección final corta. **12/12** y robusto a error de localización.

No usa Nav2 `opennav_docking` (no está en Humble). Alternativa futura si se
quiere más robustez de pose: AprilTag en vez de `cv2.QRCodeDetector`.
