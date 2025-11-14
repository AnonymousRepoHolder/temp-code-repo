# ModelVirtualizer & Original (Docker)

>  Intel Xeon Gold 6530 (2.7GHz, 32 cores), 126GB RAM, Ubuntu 24.04.3 LTS (TFLite v2.18.1, GCC 13.2.0, -O3 -march=native
>

| Model (From small to big) | Error                                                        | Latency ratio (virtualized/original)                       | Memory usage ratio(virtualized/original) |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| squeezenet                | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0149199/0.012097=123.3%<br/>0.298969/0.298861=100.0%     | 25.6484/25=102.6%                        |
| posenet                   | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output2:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output3:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.0170115/0.015065=112.9%<br/>0.177104/0.183358=96.6%      | 27/24=112.5%                             |
| fruit                     | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0161212/0.0114402=140.9%<br/>0.0445966/0.0440573=101.2%  | 18.7148/18=104.0%                        |
| lenet                     | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0155384/0.0106462=146.0%<br/>0.0100477/0.00865025=116.2% | 16.3008/16=101.9%                        |
| mobilenet                 | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.024824/0.0176189=140.9%<br/>0.0807846/0.0703794=114.8%   | 28.4922/28=101.8%                        |
| skin                      | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0234393/0.0163481=143.4%<br/>0.159943/0.122031=131.1%    | 42.75/43=99.4%                           |
| mnasnet                   | **Output0:** MSE=0.000000000000031, MAE=0.00000013, MaxAE=0.00000092, RelMAE=0.00000069<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0333578/0.0173797=191.9%<br/>0.107347/0.0849999=126.3%   | 43.1758/42=102.8%                        |
| efficientnet              | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0311206/0.0180457=172.5%<br/>0.131918/0.10964=120.3%     | 46.3047/47=98.5%                         |
| ssd                       | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.0517102/0.0294649=175.5%<br/>0.289768/0.331885=87.3%     | 80.5508/78=103.3%                        |
| depth_estimation          | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0         | 0.124427/0.108158=115.0%<br/>1.30483/1.30135=100.3%        | 163.066/163=100.0%                       |
| distilgpt2-official       | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output2:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output3:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output4:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output5:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output6:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.500856/0.134419=372.6%<br/>0.524371/0.57477=91.2%        | 630.418/627=100.5%                       |

# ModelObfuscator & Original

> Tensorflow==2.9.1，--extra_layer\=\=0, --shortcut\=\=0

| Model (From small to big) | Latency ratio (obfuscated/native) | Memory usage ratio(obfuscated/native) |
| ------------------------- | --------------------------------- | ------------------------------------- |
| squeezenet                | 0.336686/0.208427=161.5%          | 28.125/27=104.2%                      |
| posenet                   | 0.675774/0.178359=378.9%          | 23.4844/25=93.9%                      |
| fruit                     | 0.290582/0.0438961=662.0%         | 17/19=89.5%                           |
| lenet                     | 0.0165688/0.00644589=257.0%       | 9/17=52.9%                            |
| mobilenet                 | 0.362796/0.0660851=549.0%         | 27.9219/29=96.3%                      |
| skin                      | 0.515843/0.149363=345.4%          | 43.2695/44=98.3%                      |
| mnasnet                   | 0.566771/0.0745729=760.0%         | 37.5/45=83.3%                         |
| efficientnet              | 0.627571/0.108498=578.4%          | 42.7617/50=85.5%                      |
| ssd                       | 1.13219/0.279773=404.7%           | 77.9297/82=95.0%                      |
| depth_estimation          | 3.03537/0.978511=310.2%           | 165.27/170=97.2%                      |



# ModelObfuscator & Original

> Tensorflow==2.9.1，--extra_layer\=\=10, --shortcut\=\=10

| Model (From small to big) | Latency ratio (virtualized/native) | Memory usage ratio(virtualized/native) |
| ------------------------- | ---------------------------------- | -------------------------------------- |
| squeezenet                | 0.350715/0.240203=146.0%           | 29.1289/27=107.9%                      |
| posenet                   | 0.648553/0.17519=370.2%            | 24.8398/25=99.4%                       |
| fruit                     | 0.247817/0.0388322=638.2%          | 20/19=105.3%                           |
| lenet                     | 0.0117552/0.00627736=187.3%        | 11/17=64.7%                            |
| mobilenet                 | 0.437284/0.0897994=487.0%          | 31/29=106.9%                           |
| skin                      | 0.587329/0.134346=437.2%           | 45.6562/43=106.2%                      |
| mnasnet                   | 0.557823/0.0903339=617.5%          | 42.4531/45=94.3%                       |
| efficientnet              | 0.61275/0.08938=685.6%             | 44.7656/49=91.4%                       |
| ssd                       | 1.10361/0.330002=334.4%            | 80.3359/81=99.2%                       |
| depth_estimation          | 3.26683/0.947233=344.9%            | 163.41/169=96.7%                       |



# ModelObfuscator & Original

> Tensorflow==2.9.1，--extra_layer\=\=30, --shortcut\=\=30

| Model (From small to big) | Latency ratio (virtualized/native) | Memory usage ratio(virtualized/native) |
| ------------------------- | ---------------------------------- | -------------------------------------- |
| squeezenet                | 0.3324/0.209089=159.0%             | 36.1289/27=133.8%                      |
| posenet                   | 0.69815/0.146894=475.3%            | 36.4922/25=146.0%                      |
| fruit                     | 0.251041/0.0388219=646.6%          | 24/19=126.3%                           |
| lenet                     | 0.0112748/0.00620788=181.6%        | 10/17=58.8%                            |
| mobilenet                 | 0.395189/0.0800658=493.6%          | 37.8672/29=130.6%                      |
| skin                      | 0.580117/0.134435=431.5%           | 53.2188/44=121.0%                      |
| mnasnet                   | 0.595533/0.074693=797.3%           | 50.457/46=109.7%                       |
| efficientnet              | 0.669027/0.108859=614.6%           | 51.8047/50=103.6%                      |
| ssd                       | 1.30836/0.396669=329.8%            | 91.9531/81=113.5%                      |
| depth_estimation          | 3.49081/1.02239=341.4%             | 172.113/169=101.8%                     |



# ModelVirtualizer & ModelObfuscator

> --extra_layer\=\=0/10/30, --shortcut\=\=0/10/30

| Model (From small to big) | Original Model Size</br>(\*.tflite) | Virtualized Model Size</br>(v_infos.json+ params.bin) | Obfuscated Model Size</br>(obf_model.tflite) | Original Backend Size</br>(libtensorflowlite.so)             | Backend Size With Virtualization</br>(libtensorflowlite.so) | Backend Size With Obfuscation</br>(libtensorflowlite.so) |
| ------------------------- | ----------------------------------- | ----------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| squeezenet                | 4.8MB                               | 24KB+4.8MB                                            | 8.0KB/12KB/12KB                              | 5.6MB (2.18.1 Docker)</br>4.9MB (2.9.1 Docker)</br>4.4MB (2.18.1 Android) | 6.1MB (Docker)</br>4.7MB (Android)                          | 13MB/13MB/13MB                                           |
| posenet                   | 4.9MB                               | 20KB+4.9MB                                            | 8.0KB/8.0KB/12KB                             |                                                              |                                                             | 13MB/13MB/13MB                                           |
| fruit                     | 5.3MB                               | 20KB+5.3MB                                            | 8.0KB/8.0KB/12KB                             |                                                              |                                                             | 13MB/13MB/13MB                                           |
| lenet                     | 6.3MB                               | 8.0KB+6.3MB                                           | 4.0KB/4.0KB/8.0KB                            |                                                              |                                                             | 12MB/12MB/12MB                                           |
| mobilenet                 | 9.9MB                               | 20KB+9.9MB                                            | 8.0KB/8.0KB/12KB                             |                                                              |                                                             | 17MB/17MB/17MB                                           |
| skin                      | 17MB                                | 20KB+17MB                                             | 8.0KB/8.0KB/12KB                             |                                                              |                                                             | 24MB/24MB/24MB                                           |
| mnasnet                   | 17MB                                | 40KB+17MB                                             | 12KB/12KB/16KB                               |                                                              |                                                             | 27MB/27MB/28MB                                           |
| efficientnet              | 18MB                                | 36KB+18MB                                             | 12KB/12KB/16KB                               |                                                              |                                                             | 28MB/28MB/28MB                                           |
| ssd                       | 27MB                                | 44KB+27MB                                             | 16KB/16KB/16KB                               |                                                              |                                                             | 37MB/37MB/37MB                                           |
| depth_estimation          | 64MB                                | 80KB+64MB                                             | 24KB/24KB/28KB                               |                                                              |                                                             | 80MB/80MB/80MB                                           |
| distilgpt2-official       | 313MB                               | 392KB+313MB                                           | /                                            |                                                              |                                                             | /                                                        |

# ModelVirtualizer & Original (Android)

>  Redmi K50 Ultra (Snapdragon 8+ Gen 1, 3.2GHz), 12GB RAM, Android 14 (TFLite v2.18.1, NDK r25c)

| Model (From small to big) | Error                                                        | Latency ratio (virtualized/original)                       | Memory usage ratio(virtualized/original) |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| squeezenet                | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0176519/0.0132416=133.3%<br/>0.452814/0.466377=97.1%     | 25.1367/24.8906=101.0%                   |
| posenet                   | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output2:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output3:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.0149978/0.011454=130.9%<br/>0.380173/0.378816=100.4%     | 27.0078/25.7891=104.7%                   |
| fruit                     | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0133032/0.00864844=153.8%<br/>0.0885571/0.0885815=100.0% | 18.5859/17.8008=104.4%                   |
| lenet                     | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0124661/0.0103673=120.2%<br/>0.00827109/0.00862589=95.9% | 15.5898/15.4492=100.9%                   |
| mobilenet                 | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0204986/0.0140053=146.4%<br/>0.183003/0.183472=99.7%     | 29.6523/28.3203=104.7%                   |
| skin                      | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0267157/0.0172475=154.9%<br/>0.314055/0.312332=100.6%    | 42.082/41.9805=100.2%                    |
| mnasnet                   | **Output0:** MSE=0.000000000000030, MAE=0.00000013, MaxAE=0.00000083, RelMAE=0.00000070<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0304192/0.0203642=149.4%<br/>0.203455/0.204016=99.7%     | 41.1641/41.9727=98.1%                    |
| efficientnet              | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output0:** Top‑1 agreement rate=1.0 | 0.0333719/0.0195902=170.3%<br/>0.249308/0.249758=99.8%     | 45.6289/45.7539=99.7%                    |
| ssd                       | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.0470899/0.0291372=161.6%<br/>0.759491/0.758729=100.1%    | 80.0742/76.8047=104.3%                   |
| depth_estimation          | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0         | 0.111241/0.0743132=149.7%<br/>2.52468/2.52442=100.0%       | 162.25/162.098=100.1%                    |
| distilgpt2-official       | **Output0:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output1:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output2:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output3:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output4:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output5:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0<br/>**Output6:** MSE=0.0, MAE=0.0, MaxAE=0.0, RelMAE=0.0 | 0.539761/0.111775=482.9%<br/>0.398246/0.443913=89.7%       | 629.691/626.328=100.5%                   |
