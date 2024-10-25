extern "C" __global__
void apply_rule_kernel(float* neighboor, float* result, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (width - 1) * (height - 1)) {
        // Calcular los índices x, y, z
        int x = idx % (width - 1);
        int y = (idx / (width - 1)) % (height - 1);
        int z = idx / ((width - 1) * (height - 1));

        // Calcular los índices lineales para acceder a los elementos de neighboor
        int base_index = (y * width + x) * 27;
        int index_000 = base_index + 0;
        int index_001 = base_index + 1;
        int index_002 = base_index + 2;
        int index_010 = base_index + 3;
        int index_011 = base_index + 4;
        int index_012 = base_index + 5;
        int index_020 = base_index + 6;
        int index_021 = base_index + 7;
        int index_022 = base_index + 8;
        int index_100 = base_index + 9;
        int index_101 = base_index + 10;
        int index_102 = base_index + 11;
        int index_110 = base_index + 12;
        int index_111 = base_index + 13;
        int index_112 = base_index + 14;
        int index_120 = base_index + 15;
        int index_121 = base_index + 16;
        int index_122 = base_index + 17;
        int index_200 = base_index + 18;
        int index_201 = base_index + 19;
        int index_202 = base_index + 20;
        int index_210 = base_index + 21;
        int index_211 = base_index + 22;
        int index_212 = base_index + 23;
        int index_220 = base_index + 24;
        int index_221 = base_index + 25;
        int index_222 = base_index + 26;

        float rule1 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule2 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule3 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule4 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]);
            
            float rule5 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule6 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]);
            
            float rule7 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule8 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule9 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]);
            
            float rule10 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule11 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule12 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule13 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule14 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule15 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule16 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule17 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]);
            
            float rule18 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule19 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule20 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]);
            
            float rule21 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule22 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]);
            
            float rule23 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule24 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule25 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule26 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule27 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule28 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule29 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule30 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule31 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]);
            
            float rule32 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule33 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]);
            
            float rule34 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule35 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule36 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule37 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule38 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule39 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule40 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule41 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]);
            
            float rule42 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]);
            
            float rule43 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule44 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule45 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule46 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule47 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule48 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule49 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule50 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule51 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule52 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]);
            
            float rule53 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule54 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]);
            
            float rule55 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule56 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule57 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]);
            
            float rule58 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule59 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule60 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule61 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule62 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule63 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule64 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule65 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]);
            
            float rule66 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]);
            
            float rule67 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule68 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]);
            
            float rule69 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule70 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_002]);
            
            float rule71 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule72 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_102]);
            
            float rule73 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule74 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_000]), neighboor[index_101]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule75 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule76 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule77 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule78 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule79 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule80 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule81 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]);
            
            float rule82 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule83 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule84 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule85 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule86 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule87 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule88 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule89 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_002]);
            
            float rule90 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]);
            
            float rule91 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule92 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_002]);
            
            float rule93 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]);
            
            float rule94 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_000]), neighboor[index_201]), neighboor[index_202]);
            
            float rule95 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_001]), neighboor[index_002]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule96 = min(min(min(min(min(min(min(min(
            neighboor[index_000], neighboor[index_001]), neighboor[index_002]),
            neighboor[index_000]), neighboor[index_001]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule97 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule98 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule99 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule100 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]);
            
            float rule101 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule102 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]);
            
            float rule103 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule104 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule105 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]);
            
            float rule106 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule107 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule108 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule109 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule110 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule111 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule112 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule113 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]);
            
            float rule114 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule115 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule116 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]);
            
            float rule117 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule118 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]);
            
            float rule119 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule120 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule121 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule122 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule123 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule124 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule125 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule126 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule127 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule128 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule129 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]);
            
            float rule130 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule131 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule132 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule133 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule134 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule135 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule136 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule137 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_102]);
            
            float rule138 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]);
            
            float rule139 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule140 = min(min(min(min(min(min(min(min(
            neighboor[index_200], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_102]);
            
            float rule141 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_201]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]);
            
            float rule142 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_100]), neighboor[index_201]), neighboor[index_202]);
            
            float rule143 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_101]), neighboor[index_102]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
            
            float rule144 = min(min(min(min(min(min(min(min(
            neighboor[index_100], neighboor[index_101]), neighboor[index_102]),
            neighboor[index_100]), neighboor[index_101]), neighboor[index_202]),
            neighboor[index_200]), neighboor[index_201]), neighboor[index_202]);
        
        result[idx] = max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(rule1, rule2), rule3), rule4), rule5), rule6), rule7), rule8), rule9), rule10), rule11), rule12), rule13), rule14), rule15), rule16), rule17), rule18), rule19), rule20), rule21), rule22), rule23), rule24), rule25), rule26), rule27), rule28), rule29), rule30), rule31), rule32), rule33), rule34), rule35), rule36), rule37), rule38), rule39), rule40), rule41), rule42), rule43), rule44), rule45), rule46), rule47), rule48), rule49), rule50), rule51), rule52), rule53), rule54), rule55), rule56), rule57), rule58), rule59), rule60), rule61), rule62), rule63), rule64), rule65), rule66), rule67), rule68), rule69), rule70), rule71), rule72), rule73), rule74), rule75), rule76), rule77), rule78), rule79), rule80), rule81), rule82), rule83), rule84), rule85), rule86), rule87), rule88), rule89), rule90), rule91), rule92), rule93), rule94), rule95), rule96), rule97), rule98), rule99), rule100), rule101), rule102), rule103), rule104), rule105), rule106), rule107), rule108), rule109), rule110), rule111), rule112), rule113), rule114), rule115), rule116), rule117), rule118), rule119), rule120), rule121), rule122), rule123), rule124), rule125), rule126), rule127), rule128), rule129), rule130), rule131), rule132), rule133), rule134), rule135), rule136), rule137), rule138), rule139), rule140), rule141), rule142), rule143), rule144);
    }
}