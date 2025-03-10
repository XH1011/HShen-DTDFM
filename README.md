# HShen-DTDFM
Addressing domain shift in few-shot fault diagnosis with diffusion digital twin-driven feature mixing
![image](https://github.com/user-attachments/assets/8b7c8b8a-cba4-469c-a2a3-9ecdbc6a71ca)

Note that the Conditional_Diffusion code environment is tensorflow==2.10.0 and the Feature_mixing environment is pytorch==2.4.1.

Feature_mixing data is of the form:

datasets/

│

├── condition_1/

│   ├── A/

│   │   ├── A.pkl

│   ├── E/

│   │   ├── E.pkl

│   ├── F/

│   │   ├── F.pkl

│   ├── G/

│   │   ├── G.pkl

│   ├── K/

│   │   ├── K.pkl

│   ├── L/

│       ├── L.pkl

│

├── condition_2/

│   ├── A/

│   │   ├── A.pkl

│   ├── E/

│   │   ├── E.pkl

│   ├── F/

│   │   ├── F.pkl

│   ├── G/

│   │   ├── G.pkl

│   ├── K/

│   │   ├── K.pkl

│   ├── L/

│       ├── L.pkl
│
