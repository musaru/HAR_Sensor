***Sensor-Based Human Activity Recognition Based on Multi-Stream Time-Varying Features With ECA-Net Dimensionality Reduction***

# Abstract
Sensor-based datasets are extensively utilized in human-computer interaction (HCI) and medical applications due to their portability and strong privacy features. Many researchers have developed sensor-based human activity recognition (HAR) systems to increase recognition performance. However, existing systems still face challenges in achieving satisfactory performance due to insufficient time-varying features and gradient explosion issues. To address these challenges, we proposed a multi-stream temporal convolutional network (TCN)-based approach for time-varying feature extraction and feature selection to recognize human activity (HA) from sensor datasets. The proposed model effectively extracts and emphasizes the spatial-temporal features of various human activities based on a 4-stream model. Each stream uses TCN to extract time-varying features and enhances them using an appropriate integration module. The first stream extracts fine-grained temporal features with TCN. The second and third streams integrate TCN features with LSTM, applying pre-integration and post-integration, respectively. The fourth stream uses CNN for spatial features and TCN for enhancing temporal features. The concatenation of the 4-stream features captures complex dependencies, improving the model’s understanding of prolonged activities. In addition, we proposed a modified effective channel attention network (ECA-Net) that assigns higher dimensionality weight to lower dimensionality, enabling the proposed model to learn and recognise human activities effectively despite their complex patterns. Evaluations on the WISDM, PAMAP2, USC-HAD, Opportunity UCI, and UCI-HAR datasets showed accuracy improvements of 1.12%, 1.99%, 1.30%, 5.72%, and 0.38%, respectively, over state-of-the-art systems. The high-performance accuracy of the proposed model demonstrates its superiority, with implications for improving prosthetic limb functionality and advancing robotics human-machine interfaces. Our data preprocessing approach, deep learning model code, and dataset information are available at the following link: https://github.com/musaru/HAR_Sensor.
# Paper link: [(https://ieeexplore.ieee.org/document/10704663)]
![image](https://github.com/user-attachments/assets/8e8f837a-e83d-4d3a-971a-ea94a002c552)

# Keywords:
Sensor, vision, gesture recognition, signal processing, human activity recognition (HAR),
convolutional neural network (CNN), temporal convolutional network (TCN), LSTM, HAR, enahance
channel attention network (ECANet).
# Dataset
![image](https://github.com/user-attachments/assets/191e03da-0e10-4b44-b1cc-c20a37a6addf)

THE WISDM DATASET presented in this study are openly available in: 
PAMAP2 DATASET presented in this study are openly available in:  
USC-HAD DATASET
OPPORTUNITY UCI DATASET presented in this study are openly available in: [(https://archive.ics.uci.edu/dataset/226/opportunity +activity+
recognition)].

DAPHNET GAIT PARKINSON DATASET [(https://archive.ics.uci.edu/dataset/245/ daphnet+
freezing+ of+gait)]
The Daphnet Gait Parkinson Dataset is a valuable
UCI HAR DATASET: [(https://archive.ics.uci.edu/dataset/240/human+activity
+recognition+using+smartphones)]
