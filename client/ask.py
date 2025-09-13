import numpy as np
from init import ask_A1
from init import ask_B1_dispersion, ask_B1_vF, ask_B2_dispersion, ask_B2_vF, ask_B3_dispersion, ask_B3_vF, ask_B4_dispersion, ask_B4_bbE, ask_B5, ask_B6
from init import ask_C1, ask_C2, ask_C3, ask_C4, ask_C5
from init import ask_D1, ask_D2, ask_D3
from init import ask_E1, ask_E2, ask_E3, ask_E4
from init import write_to_text

from call_server import run, status, cancel
from call_db import save_all_results, get_attempts_left





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Generate prompts -------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_A_prompts(resolution, size):

    print("Q1")
    Q1, L1 = ask_A1(resolution, size, 0); Q2, L2 = ask_A1(resolution, size, 0.2); Q3, L3 = ask_A1(resolution, size, 0.4); Q4, L4 = ask_A1(resolution, size, 0.6); Q5, L5 = ask_A1(resolution, size, 0.8)
    questions = [Q1, Q2, Q3, Q4, Q5]
    lengths = [L1, L2, L3, L4, L5]
    titles = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    return questions, lengths, titles



def retrieve_A_prompts(resolution, size):

    if size == 1:
        file_path = f"Prompts_small/A1/A1_r{resolution}_n"
    elif size == 2:
        file_path = f"Prompts_med/A1/A1_r{resolution}_n"
    else:
        file_path = f"Prompts_large/A1/A1_r{resolution}_n"
    
    with open(file_path + "0_Q.txt", 'r') as file:
        Q1 = file.read()
    with open(file_path + "20_Q.txt", 'r') as file:
        Q2 = file.read()
    with open(file_path + "40_Q.txt", 'r') as file:
        Q3 = file.read()
    with open(file_path + "60_Q.txt", 'r') as file:
        Q4 = file.read()
    with open(file_path + "80_Q.txt", 'r') as file:
        Q5 = file.read()
    
    questions = [Q1, Q2, Q3, Q4, Q5]
    lengths = [0, 0, 0, 0, 0]
    titles = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    return questions, lengths, titles



def get_A_titles():
    titles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    return titles



def get_BCD_prompts(resolution, size):

    Q6, L6 = ask_B1_dispersion(resolution, size, 0); Q7, L7 = ask_B1_dispersion(resolution, size, 0.1); Q8, L8 = ask_B1_dispersion(resolution, size, 0.2); Q9, L9 = ask_B1_dispersion(resolution, size, 0.3); print("Q10"); Q10, L10 = ask_B1_dispersion(resolution, size, 0.4)
    Q11, L11 = ask_B1_vF(resolution, size, 0); Q12, L12 = ask_B1_vF(resolution, size, 0.1); Q13, L13 = ask_B1_vF(resolution, size, 0.2); Q14, L14 = ask_B1_vF(resolution, size, 0.3); Q15, L15 = ask_B1_vF(resolution, size, 0.4)

    Q16, L16 = ask_B2_dispersion(resolution, size, 0); Q17, L17 = ask_B2_dispersion(resolution, size, 0.1); Q18, L18 = ask_B2_dispersion(resolution, size, 0.2); Q19, L19 = ask_B2_dispersion(resolution, size, 0.3); print("Q20"); Q20, L20 = ask_B2_dispersion(resolution, size, 0.4)
    Q21, L21 = ask_B2_vF(resolution, size, 0); Q22, L22 = ask_B2_vF(resolution, size, 0.1); Q23, L23 = ask_B2_vF(resolution, size, 0.2); Q24, L24 = ask_B2_vF(resolution, size, 0.3); Q25, L25 = ask_B2_vF(resolution, size, 0.4)

    Q26, L26 = ask_B3_dispersion(resolution, size, 0); Q27, L27 = ask_B3_dispersion(resolution, size, 0.1); Q28, L28 = ask_B3_dispersion(resolution, size, 0.2); Q29, L29 = ask_B3_dispersion(resolution, size, 0.3); print("Q30"); Q30, L30 = ask_B3_dispersion(resolution, size, 0.4)
    Q31, L31 = ask_B3_vF(resolution, size, 0); Q32, L32 = ask_B3_vF(resolution, size, 0.1); Q33, L33 = ask_B3_vF(resolution, size, 0.2); Q34, L34 = ask_B3_vF(resolution, size, 0.3); Q35, L35 = ask_B3_vF(resolution, size, 0.4)

    Q36, L36 = ask_B4_dispersion(resolution, size, 0); Q37, L37 = ask_B4_dispersion(resolution, size, 0.1); Q38, L38 = ask_B4_dispersion(resolution, size, 0.2); Q39, L39 = ask_B4_dispersion(resolution, size, 0.3); print("Q40"); Q40, L40 = ask_B4_dispersion(resolution, size, 0.4)
    Q41, L41 = ask_B4_bbE(resolution, size, 0); Q42, L42 = ask_B4_bbE(resolution, size, 0.1); Q43, L43 = ask_B4_bbE(resolution, size, 0.2); Q44, L44 = ask_B4_bbE(resolution, size, 0.3); Q45, L45 = ask_B4_bbE(resolution, size, 0.4)

    Q46, L46 = ask_B5(resolution, size, 0); Q47, L47 = ask_B5(resolution, size, 0.1); Q48, L48 = ask_B5(resolution, size, 0.2); Q49, L49 = ask_B5(resolution, size, 0.3); print("Q50"); Q50, L50 = ask_B5(resolution, size, 0.4)
    Q51, L51 = ask_B6(resolution, size, 0); Q52, L52 = ask_B6(resolution, size, 0.1); Q53, L53 = ask_B6(resolution, size, 0.2); Q54, L54 = ask_B6(resolution, size, 0.3); Q55, L55 = ask_B6(resolution, size, 0.4)

    Q56, L56 = ask_C1(resolution, size, 0); Q57, L57 = ask_C1(resolution, size, 0.05); Q58, L58 = ask_C1(resolution, size, 0.1); Q59, L59 = ask_C1(resolution, size, 0.15); print("Q60"); Q60, L60 = ask_C1(resolution, size, 0.2)
    Q61, L61 = ask_C2(resolution, size, 0); Q62, L62 = ask_C2(resolution, size, 0.05); Q63, L63 = ask_C2(resolution, size, 0.1); Q64, L64 = ask_C2(resolution, size, 0.15); Q65, L65 = ask_C2(resolution, size, 0.2)
    Q66, L66 = ask_C3(resolution, size, 0); Q67, L67 = ask_C3(resolution, size, 0.05); Q68, L68 = ask_C3(resolution, size, 0.1); Q69, L69 = ask_C3(resolution, size, 0.15); print("Q70"); Q70, L70 = ask_C3(resolution, size, 0.2)
    Q71, L71 = ask_C4(resolution, size, 0); Q72, L72 = ask_C4(resolution, size, 0.05); Q73, L73 = ask_C4(resolution, size, 0.1); Q74, L74 = ask_C4(resolution, size, 0.15); Q75, L75 = ask_C4(resolution, size, 0.2)
    Q76, L76 = ask_C5(resolution, size, 0); Q77, L77 = ask_C5(resolution, size, 0.05); Q78, L78 = ask_C5(resolution, size, 0.1); Q79, L79 = ask_C5(resolution, size, 0.15); print("Q80"); Q80, L80 = ask_C5(resolution, size, 0.2)

    Q81, L81 = ask_D1(resolution, size, 0, 0.5); Q82, L82 = ask_D1(resolution, size, 0.1, 0.5); Q83, L83 = ask_D1(resolution, size, 0.2, 0.5); Q84, L84 = ask_D1(resolution, size, 0.3, 0.5); Q85, L85 = ask_D1(resolution, size, 0.4, 0.5)
    Q86, L86 = ask_D1(resolution, size, 0, 0.75); Q87, L87 = ask_D1(resolution, size, 0.1, 0.75); Q88, L88 = ask_D1(resolution, size, 0.2, 0.75); Q89, L89 = ask_D1(resolution, size, 0.3, 0.75); print("Q90"); Q90, L90 = ask_D1(resolution, size, 0.4, 0.75)
    Q91, L91 = ask_D1(resolution, size, 0, 1); Q92, L92 = ask_D1(resolution, size, 0.1, 1); Q93, L93 = ask_D1(resolution, size, 0.2, 1); Q94, L94 = ask_D1(resolution, size, 0.3, 1); Q95, L95 = ask_D1(resolution, size, 0.4, 1)
    Q96, L96 = ask_D1(resolution, size, 0, 2); Q97, L97 = ask_D1(resolution, size, 0.1, 2); Q98, L98 = ask_D1(resolution, size, 0.2, 2); Q99, L99 = ask_D1(resolution, size, 0.3, 2); print("Q100"); Q100, L100 = ask_D1(resolution, size, 0.4, 2)
    Q101, L101 = ask_D1(resolution, size, 0, 5); Q102, L102 = ask_D1(resolution, size, 0.1, 5); Q103, L103 = ask_D1(resolution, size, 0.2, 5); Q104, L104 = ask_D1(resolution, size, 0.3, 5); Q105, L105 = ask_D1(resolution, size, 0.4, 5)

    Q106, L106 = ask_D2(resolution, size, 0); Q107, L107 = ask_D2(resolution, size, 0.1); Q108, L108 = ask_D2(resolution, size, 0.2); Q109, L109 = ask_D2(resolution, size, 0.3); print("Q110"); Q110, L110 = ask_D2(resolution, size, 0.4)
    Q111, L111 = ask_D3(resolution, size, 0); Q112, L112 = ask_D3(resolution, size, 0.1); Q113, L113 = ask_D3(resolution, size, 0.2); Q114, L114 = ask_D3(resolution, size, 0.3); Q115, L115 = ask_D3(resolution, size, 0.4)

    questions_1 = [Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34, Q35, Q36, Q37, Q38, Q39, Q40]
    questions_2 = [Q41, Q42, Q43, Q44, Q45, Q46, Q47, Q48, Q49, Q50, Q51, Q52, Q53, Q54, Q55, Q56, Q57, Q58, Q59, Q60, Q61, Q62, Q63, Q64, Q65, Q66, Q67, Q68, Q69, Q70]
    questions_3 = [Q71, Q72, Q73, Q74, Q75, Q76, Q77, Q78, Q79, Q80, Q81, Q82, Q83, Q84, Q85, Q86, Q87, Q88, Q89, Q90, Q91, Q92, Q93, Q94, Q95, Q96, Q97, Q98, Q99, Q100]
    questions_4 = [Q101, Q102, Q103, Q104, Q105, Q106, Q107, Q108, Q109, Q110, Q111, Q112, Q113, Q114, Q115]
    questions = [item for item in questions_1] + [item for item in questions_2] + [item for item in questions_3] + [item for item in questions_4]

    lengths_1 = [L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24, L25, L26, L27, L28, L29, L30, L31, L32, L33, L34, L35, L36, L37, L38, L39, L40]
    lengths_2 = [L41, L42, L43, L44, L45, L46, L47, L48, L49, L50, L51, L52, L53, L54, L55, L56, L57, L58, L59, L60, L61, L62, L63, L64, L65, L66, L67, L68, L69, L70]
    lengths_3 = [L71, L72, L73, L74, L75, L76, L77, L78, L79, L80, L81, L82, L83, L84, L85, L86, L87, L88, L89, L90, L91, L92, L93, L94, L95, L96, L97, L98, L99, L100]
    lengths_4 = [L101, L102, L103, L104, L105, L106, L107, L108, L109, L110, L111, L112, L113, L114, L115]
    lengths = [item for item in lengths_1] + [item for item in lengths_2] + [item for item in lengths_3] + [item for item in lengths_4]

    titles_1 = ["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40"]
    titles_2 = ["Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70"]
    titles_3 = ["Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100"]
    titles_4 = ["Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115"]
    titles = [item for item in titles_1] + [item for item in titles_2] + [item for item in titles_3] + [item for item in titles_4]

    return questions, lengths, titles



def retrieve_BCD_prompts(resolution, size):

    if size == 1:
        size_str = "small"
    elif size == 2:
        size_str = "med"
    else:
        size_str = "large"
    
    file_path_B1 = "Prompts_" + size_str + f"/B1/B1_r{resolution}_n"
    file_path_B1_vF = "Prompts_" + size_str + f"/B1/B1_vF_r{resolution}_n"
    file_path_B2 = "Prompts_" + size_str + f"/B2/B2_r{resolution}_n"
    file_path_B2_vF = "Prompts_" + size_str + f"/B2/B2_vF_r{resolution}_n"
    file_path_B3 = "Prompts_" + size_str + f"/B3/B3_r{resolution}_n"
    file_path_B3_vF = "Prompts_" + size_str + f"/B3/B3_vF_r{resolution}_n"
    file_path_B4 = "Prompts_" + size_str + f"/B4/B4_r{resolution}_n"
    file_path_B4_bbE = "Prompts_" + size_str + f"/B4/B4_bbE_r{resolution}_n"
    file_path_B5 = "Prompts_" + size_str + f"/B5/B5_r{resolution}_n"
    file_path_B6 = "Prompts_" + size_str + f"/B6/B6_r{resolution}_n"
    file_path_C1 = "Prompts_" + size_str + f"/C1/C1_r{resolution}_n"
    file_path_C2 = "Prompts_" + size_str + f"/C2/C2_r{resolution}_n"
    file_path_C3 = "Prompts_" + size_str + f"/C3/C3_r{resolution}_n"
    file_path_C4 = "Prompts_" + size_str + f"/C4/C4_r{resolution}_n"
    file_path_C5 = "Prompts_" + size_str + f"/C5/C5_r{resolution}_n"
    file_path_D1_L05 = "Prompts_" + size_str + f"/D1/D1_L05_r{resolution}_n"
    file_path_D1_L075 = "Prompts_" + size_str + f"/D1/D1_L075_r{resolution}_n"
    file_path_D1_L1 = "Prompts_" + size_str + f"/D1/D1_L1_r{resolution}_n"
    file_path_D1_L2 = "Prompts_" + size_str + f"/D1/D1_L2_r{resolution}_n"
    file_path_D1_L5 = "Prompts_" + size_str + f"/D1/D1_L5_r{resolution}_n"
    file_path_D2 = "Prompts_" + size_str + f"/D2/D2_r{resolution}_n"
    file_path_D3 = "Prompts_" + size_str + f"/D3/D3_r{resolution}_n"

    Q6, L6 = ask_B1_dispersion(resolution, size, 0)
    
    with open(file_path_B1 + "0_Q.txt", 'r') as file:
        Q6 = file.read()
    with open(file_path_B1 + "10_Q.txt", 'r') as file:
        Q7 = file.read()
    with open(file_path_B1 + "20_Q.txt", 'r') as file:
        Q8 = file.read()
    with open(file_path_B1 + "30_Q.txt", 'r') as file:
        Q9 = file.read()
    with open(file_path_B1 + "40_Q.txt", 'r') as file:
        Q10 = file.read()
    
    with open(file_path_B1_vF + "0_Q.txt", 'r') as file:
        Q11 = file.read()
    with open(file_path_B1_vF + "10_Q.txt", 'r') as file:
        Q12 = file.read()
    with open(file_path_B1_vF + "20_Q.txt", 'r') as file:
        Q13 = file.read()
    with open(file_path_B1_vF + "30_Q.txt", 'r') as file:
        Q14 = file.read()
    with open(file_path_B1_vF + "40_Q.txt", 'r') as file:
        Q15 = file.read()
    
    with open(file_path_B2 + "0_Q.txt", 'r') as file:
        Q16 = file.read()
    with open(file_path_B2 + "10_Q.txt", 'r') as file:
        Q17 = file.read()
    with open(file_path_B2 + "20_Q.txt", 'r') as file:
        Q18 = file.read()
    with open(file_path_B2 + "30_Q.txt", 'r') as file:
        Q19 = file.read()
    with open(file_path_B2 + "40_Q.txt", 'r') as file:
        Q20 = file.read()
    
    with open(file_path_B2_vF + "0_Q.txt", 'r') as file:
        Q21 = file.read()
    with open(file_path_B2_vF + "10_Q.txt", 'r') as file:
        Q22 = file.read()
    with open(file_path_B2_vF + "20_Q.txt", 'r') as file:
        Q23 = file.read()
    with open(file_path_B2_vF + "30_Q.txt", 'r') as file:
        Q24 = file.read()
    with open(file_path_B2_vF + "40_Q.txt", 'r') as file:
        Q25 = file.read()
    
    with open(file_path_B3 + "0_Q.txt", 'r') as file:
        Q26 = file.read()
    with open(file_path_B3 + "10_Q.txt", 'r') as file:
        Q27 = file.read()
    with open(file_path_B3 + "20_Q.txt", 'r') as file:
        Q28 = file.read()
    with open(file_path_B3 + "30_Q.txt", 'r') as file:
        Q29 = file.read()
    with open(file_path_B3 + "40_Q.txt", 'r') as file:
        Q30 = file.read()
    
    with open(file_path_B3_vF + "0_Q.txt", 'r') as file:
        Q31 = file.read()
    with open(file_path_B3_vF + "10_Q.txt", 'r') as file:
        Q32 = file.read()
    with open(file_path_B3_vF + "20_Q.txt", 'r') as file:
        Q33 = file.read()
    with open(file_path_B3_vF + "30_Q.txt", 'r') as file:
        Q34 = file.read()
    with open(file_path_B3_vF + "40_Q.txt", 'r') as file:
        Q35 = file.read()
    
    with open(file_path_B4 + "0_Q.txt", 'r') as file:
        Q36 = file.read()
    with open(file_path_B4 + "10_Q.txt", 'r') as file:
        Q37 = file.read()
    with open(file_path_B4 + "20_Q.txt", 'r') as file:
        Q38 = file.read()
    with open(file_path_B4 + "30_Q.txt", 'r') as file:
        Q39 = file.read()
    with open(file_path_B4 + "40_Q.txt", 'r') as file:
        Q40 = file.read()
    
    with open(file_path_B4_bbE + "0_Q.txt", 'r') as file:
        Q41 = file.read()
    with open(file_path_B4_bbE + "10_Q.txt", 'r') as file:
        Q42 = file.read()
    with open(file_path_B4_bbE + "20_Q.txt", 'r') as file:
        Q43 = file.read()
    with open(file_path_B4_bbE + "30_Q.txt", 'r') as file:
        Q44 = file.read()
    with open(file_path_B4_bbE + "40_Q.txt", 'r') as file:
        Q45 = file.read()
    
    with open(file_path_B5 + "0_Q.txt", 'r') as file:
        Q46 = file.read()
    with open(file_path_B5 + "10_Q.txt", 'r') as file:
        Q47 = file.read()
    with open(file_path_B5 + "20_Q.txt", 'r') as file:
        Q48 = file.read()
    with open(file_path_B5 + "30_Q.txt", 'r') as file:
        Q49 = file.read()
    with open(file_path_B5 + "40_Q.txt", 'r') as file:
        Q50 = file.read()
    
    with open(file_path_B6 + "0_Q.txt", 'r') as file:
        Q51 = file.read()
    with open(file_path_B6 + "10_Q.txt", 'r') as file:
        Q52 = file.read()
    with open(file_path_B6 + "20_Q.txt", 'r') as file:
        Q53 = file.read()
    with open(file_path_B6 + "30_Q.txt", 'r') as file:
        Q54 = file.read()
    with open(file_path_B6 + "40_Q.txt", 'r') as file:
        Q55 = file.read()
    
    with open(file_path_C1 + "0_Q.txt", 'r') as file:
        Q56 = file.read()
    with open(file_path_C1 + "5_Q.txt", 'r') as file:
        Q57 = file.read()
    with open(file_path_C1 + "10_Q.txt", 'r') as file:
        Q58 = file.read()
    with open(file_path_C1 + "15_Q.txt", 'r') as file:
        Q59 = file.read()
    with open(file_path_C1 + "20_Q.txt", 'r') as file:
        Q60 = file.read()
    
    with open(file_path_C2 + "0_Q.txt", 'r') as file:
        Q61 = file.read()
    with open(file_path_C2 + "5_Q.txt", 'r') as file:
        Q62 = file.read()
    with open(file_path_C2 + "10_Q.txt", 'r') as file:
        Q63 = file.read()
    with open(file_path_C2 + "15_Q.txt", 'r') as file:
        Q64 = file.read()
    with open(file_path_C2 + "20_Q.txt", 'r') as file:
        Q65 = file.read()
    
    with open(file_path_C3 + "0_Q.txt", 'r') as file:
        Q66 = file.read()
    with open(file_path_C3 + "5_Q.txt", 'r') as file:
        Q67 = file.read()
    with open(file_path_C3 + "10_Q.txt", 'r') as file:
        Q68 = file.read()
    with open(file_path_C3 + "15_Q.txt", 'r') as file:
        Q69 = file.read()
    with open(file_path_C3 + "20_Q.txt", 'r') as file:
        Q70 = file.read()
    
    with open(file_path_C4 + "0_Q.txt", 'r') as file:
        Q71 = file.read()
    with open(file_path_C4 + "5_Q.txt", 'r') as file:
        Q72 = file.read()
    with open(file_path_C4 + "10_Q.txt", 'r') as file:
        Q73 = file.read()
    with open(file_path_C4 + "15_Q.txt", 'r') as file:
        Q74 = file.read()
    with open(file_path_C4 + "20_Q.txt", 'r') as file:
        Q75 = file.read()
    
    with open(file_path_C5 + "0_Q.txt", 'r') as file:
        Q76 = file.read()
    with open(file_path_C5 + "5_Q.txt", 'r') as file:
        Q77 = file.read()
    with open(file_path_C5 + "10_Q.txt", 'r') as file:
        Q78 = file.read()
    with open(file_path_C5 + "15_Q.txt", 'r') as file:
        Q79 = file.read()
    with open(file_path_C5 + "20_Q.txt", 'r') as file:
        Q80 = file.read()
    
    with open(file_path_D1_L05 + "0_Q.txt", 'r') as file:
        Q81 = file.read()
    with open(file_path_D1_L05 + "10_Q.txt", 'r') as file:
        Q82 = file.read()
    with open(file_path_D1_L05 + "20_Q.txt", 'r') as file:
        Q83 = file.read()
    with open(file_path_D1_L05 + "30_Q.txt", 'r') as file:
        Q84 = file.read()
    with open(file_path_D1_L05 + "40_Q.txt", 'r') as file:
        Q85 = file.read()
    
    with open(file_path_D1_L075 + "0_Q.txt", 'r') as file:
        Q86 = file.read()
    with open(file_path_D1_L075 + "10_Q.txt", 'r') as file:
        Q87 = file.read()
    with open(file_path_D1_L075 + "20_Q.txt", 'r') as file:
        Q88 = file.read()
    with open(file_path_D1_L075 + "30_Q.txt", 'r') as file:
        Q89 = file.read()
    with open(file_path_D1_L075 + "40_Q.txt", 'r') as file:
        Q90 = file.read()
    
    with open(file_path_D1_L1 + "0_Q.txt", 'r') as file:
        Q91 = file.read()
    with open(file_path_D1_L1 + "10_Q.txt", 'r') as file:
        Q92 = file.read()
    with open(file_path_D1_L1 + "20_Q.txt", 'r') as file:
        Q93 = file.read()
    with open(file_path_D1_L1 + "30_Q.txt", 'r') as file:
        Q94 = file.read()
    with open(file_path_D1_L1 + "40_Q.txt", 'r') as file:
        Q95 = file.read()
    
    with open(file_path_D1_L2 + "0_Q.txt", 'r') as file:
        Q96 = file.read()
    with open(file_path_D1_L2 + "10_Q.txt", 'r') as file:
        Q97 = file.read()
    with open(file_path_D1_L2 + "20_Q.txt", 'r') as file:
        Q98 = file.read()
    with open(file_path_D1_L2 + "30_Q.txt", 'r') as file:
        Q99 = file.read()
    with open(file_path_D1_L2 + "40_Q.txt", 'r') as file:
        Q100 = file.read()
    
    with open(file_path_D1_L5 + "0_Q.txt", 'r') as file:
        Q101 = file.read()
    with open(file_path_D1_L5 + "10_Q.txt", 'r') as file:
        Q102 = file.read()
    with open(file_path_D1_L5 + "20_Q.txt", 'r') as file:
        Q103 = file.read()
    with open(file_path_D1_L5 + "30_Q.txt", 'r') as file:
        Q104 = file.read()
    with open(file_path_D1_L5 + "40_Q.txt", 'r') as file:
        Q105 = file.read()
    
    with open(file_path_D2 + "0_Q.txt", 'r') as file:
        Q106 = file.read()
    with open(file_path_D2 + "10_Q.txt", 'r') as file:
        Q107 = file.read()
    with open(file_path_D2 + "20_Q.txt", 'r') as file:
        Q108 = file.read()
    with open(file_path_D2 + "30_Q.txt", 'r') as file:
        Q109 = file.read()
    with open(file_path_D2 + "40_Q.txt", 'r') as file:
        Q110 = file.read()
    
    with open(file_path_D3 + "0_Q.txt", 'r') as file:
        Q111 = file.read()
    with open(file_path_D3 + "10_Q.txt", 'r') as file:
        Q112 = file.read()
    with open(file_path_D3 + "20_Q.txt", 'r') as file:
        Q113 = file.read()
    with open(file_path_D3 + "30_Q.txt", 'r') as file:
        Q114 = file.read()
    with open(file_path_D3 + "40_Q.txt", 'r') as file:
        Q115 = file.read()

    questions_1 = [Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34, Q35, Q36, Q37, Q38, Q39, Q40]
    questions_2 = [Q41, Q42, Q43, Q44, Q45, Q46, Q47, Q48, Q49, Q50, Q51, Q52, Q53, Q54, Q55, Q56, Q57, Q58, Q59, Q60, Q61, Q62, Q63, Q64, Q65, Q66, Q67, Q68, Q69, Q70]
    questions_3 = [Q71, Q72, Q73, Q74, Q75, Q76, Q77, Q78, Q79, Q80, Q81, Q82, Q83, Q84, Q85, Q86, Q87, Q88, Q89, Q90, Q91, Q92, Q93, Q94, Q95, Q96, Q97, Q98, Q99, Q100]
    questions_4 = [Q101, Q102, Q103, Q104, Q105, Q106, Q107, Q108, Q109, Q110, Q111, Q112, Q113, Q114, Q115]
    questions = [item for item in questions_1] + [item for item in questions_2] + [item for item in questions_3] + [item for item in questions_4]

    lengths_1 = [L6, L6, L6, L6, L6, 0, 0, 0, 0, 0, L6, L6, L6, L6, L6, 0, 0, 0, 0, 0, L6, L6, L6, L6, L6, 0, 0, 0, 0, 0, resolution, resolution, resolution, resolution, resolution]
    lengths_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, L6]
    lengths_3 = [L6, L6, L6, L6, L6, L6, L6, L6, L6, L6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lengths_4 = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    lengths = [item for item in lengths_1] + [item for item in lengths_2] + [item for item in lengths_3] + [item for item in lengths_4]

    titles_1 = ["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40"]
    titles_2 = ["Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70"]
    titles_3 = ["Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100"]
    titles_4 = ["Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115"]
    titles = [item for item in titles_1] + [item for item in titles_2] + [item for item in titles_3] + [item for item in titles_4]

    return questions, lengths, titles



def get_BCD_titles():
    titles_1 = ["Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40"]
    titles_2 = ["Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70"]
    titles_3 = ["Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100"]
    titles_4 = ["Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115"]
    titles = [item for item in titles_1] + [item for item in titles_2] + [item for item in titles_3] + [item for item in titles_4]
    return titles



def get_E_prompts(resolution_1, resolution, size):

    Q116, L116 = ask_E1(resolution_1, size, 0); Q117, L117 = ask_E1(resolution_1, size, 0.1); Q118, L118 = ask_E1(resolution_1, size, 0.2); Q119, L119 = ask_E1(resolution_1, size, 0.3); print("Q120"); Q120, L120 = ask_E1(resolution_1, size, 0.4)
    Q121, L121 = ask_E2(resolution, size, 0); Q122, L122 = ask_E2(resolution, size, 0.1); Q123, L123 = ask_E2(resolution, size, 0.2); Q124, L124 = ask_E2(resolution, size, 0.3); Q125, L125 = ask_E2(resolution, size, 0.4)
    Q126, L126 = ask_E3(resolution, size, 0); Q127, L127 = ask_E3(resolution, size, 0.1); Q128, L128 = ask_E3(resolution, size, 0.2); Q129, L129 = ask_E3(resolution, size, 0.3); print("Q130"); Q130, L130 = ask_E3(resolution, size, 0.4)
    Q131, L131 = ask_E4(resolution, size, 0); Q132, L132 = ask_E4(resolution, size, 0.1); Q133, L133 = ask_E4(resolution, size, 0.2); Q134, L134 = ask_E4(resolution, size, 0.3); Q135, L135 = ask_E4(resolution, size, 0.4)

    questions = [Q116, Q117, Q118, Q119, Q120, Q121, Q122, Q123, Q124, Q125, Q126, Q127, Q128, Q129, Q130, Q131, Q132, Q133, Q134, Q135]
    lengths = [L116, L117, L118, L119, L120, L121, L122, L123, L124, L125, L126, L127, L128, L129, L130, L131, L132, L133, L134, L135]
    titles = ["Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135"]

    return questions, lengths, titles



def retrieve_E_prompts(resolution_1, resolution, size):

    if size == 1:
        size_str = "small"
    elif size == 2:
        size_str = "med"
    else:
        size_str = "large"
    
    file_path_E1 = "Prompts_" + size_str + f"/E1/E1_r{resolution_1}_n"
    file_path_E2 = "Prompts_" + size_str + f"/E2/E2_r{resolution}_n"
    file_path_E3 = "Prompts_" + size_str + f"/E3/E3_r{resolution}_n"
    file_path_E4 = "Prompts_" + size_str + f"/E4/E4_r{resolution}_n"

    with open(file_path_E1 + "0_Q.txt", 'r') as file:
        Q116 = file.read()
    with open(file_path_E1 + "10_Q.txt", 'r') as file:
        Q117 = file.read()
    with open(file_path_E1 + "20_Q.txt", 'r') as file:
        Q118 = file.read()
    with open(file_path_E1 + "30_Q.txt", 'r') as file:
        Q119 = file.read()
    with open(file_path_E1 + "40_Q.txt", 'r') as file:
        Q120 = file.read()

    with open(file_path_E2 + "0_Q.txt", 'r') as file:
        Q121 = file.read()
    with open(file_path_E2 + "10_Q.txt", 'r') as file:
        Q122 = file.read()
    with open(file_path_E2 + "20_Q.txt", 'r') as file:
        Q123 = file.read()
    with open(file_path_E2 + "30_Q.txt", 'r') as file:
        Q124 = file.read()
    with open(file_path_E2 + "40_Q.txt", 'r') as file:
        Q125 = file.read()

    with open(file_path_E3 + "0_Q.txt", 'r') as file:
        Q126 = file.read()
    with open(file_path_E3 + "10_Q.txt", 'r') as file:
        Q127 = file.read()
    with open(file_path_E3 + "20_Q.txt", 'r') as file:
        Q128 = file.read()
    with open(file_path_E3 + "30_Q.txt", 'r') as file:
        Q129 = file.read()
    with open(file_path_E3 + "40_Q.txt", 'r') as file:
        Q130 = file.read()

    with open(file_path_E4 + "0_Q.txt", 'r') as file:
        Q131 = file.read()
    with open(file_path_E4 + "10_Q.txt", 'r') as file:
        Q132 = file.read()
    with open(file_path_E4 + "20_Q.txt", 'r') as file:
        Q133 = file.read()
    with open(file_path_E4 + "30_Q.txt", 'r') as file:
        Q134 = file.read()
    with open(file_path_E4 + "40_Q.txt", 'r') as file:
        Q135 = file.read()

    questions = [Q116, Q117, Q118, Q119, Q120, Q121, Q122, Q123, Q124, Q125, Q126, Q127, Q128, Q129, Q130, Q131, Q132, Q133, Q134, Q135]
    lengths = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    titles = ["Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135"]

    return questions, lengths, titles



def get_E_titles():
    titles = ["Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135"]
    return titles





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Ask as a batch of models and prompts -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------- List of resolutions (small/med/large) ----------------------------------
# ---                                                                                                     ---
# ---                                   A1:        75 / 125 / 250                                         ---
# ---                                   B1---D3:   100 / 200 / 300                                        ---
# ---                                   E1:        80 / 150 / 250                                         ---
# ---                                   E2---E4:   110 / 220 / 350                                        ---
# ---                                                                                                     ---
# -----------------------------------------------------------------------------------------------------------

def ask_batch(size, retrieve, attempts):

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
    # "gemini-2.5-pro-preview-03-25",
    # "claude-3-7-sonnet-20250219",
    # "o3-pro-2025-06-10",
    # "grok-4",
    "kimi-k2",
    "glm-4.5",
    "qwen3-235b-thinking",
    "gpt-oss-20b",
    "gpt-oss-120b"
    ]

    titles_A = get_A_titles()
    titles_BCD = get_BCD_titles()
    titles_E = get_E_titles()
    titles = titles_A + titles_BCD + titles_E

    if size == 1:
        save_dir = "Responses_small"
        prompts_str = "Prompts_small/"
    elif size == 2:
        save_dir = "Responses_med"
        prompts_str = "Prompts_med/"
        titles = [s + "_med" for s in titles]
    else:
        save_dir = "Responses_large"
        prompts_str = "Prompts_large/"
        titles = [s + "_large" for s in titles]
    
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)

    # Questions and lengths
    if size == 1:
        resolution_A1 = 75
        resolution_BCD = 100
        resolution_E1 = 80
        resolution_E234 = 110
    elif size == 2:
        resolution_A1 = 125
        resolution_BCD = 200
        resolution_E1 = 150
        resolution_E234 = 220
    else:
        resolution_A1 = 250
        resolution_BCD = 300
        resolution_E1 = 250
        resolution_E234 = 350

    if retrieve == 0:
        questions_A, lengths_A, titles_A = get_A_prompts(resolution_A1, size)
        questions_BCD, lengths_BCD, titles_BCD = get_BCD_prompts(resolution_BCD, size)
        questions_E, lengths_E, titles_E = get_E_prompts(resolution_E1, resolution_E234, size)
    else:
        questions_A, lengths_A, titles_A = retrieve_A_prompts(resolution_A1, size)
        questions_BCD, lengths_BCD, titles_BCD = retrieve_BCD_prompts(resolution_BCD, size)
        questions_E, lengths_E, titles_E = retrieve_E_prompts(resolution_E1, resolution_E234, size)

    prompts = questions_A + questions_BCD + questions_E
    num_elems = lengths_A + lengths_BCD + lengths_E
    print("Response lengths =", num_elems)

    write_to_text(str(prompts), prompts_str, "Prompts_all")
    run(models, prompts, titles, num_elems, num_repeats = attempts_left)

    return



def save_batch(size, attempts):

    titles_A = get_A_titles()
    titles_BCD = get_BCD_titles()
    titles_E = get_E_titles()

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
    # "gemini-2.5-pro-preview-03-25",
    # "claude-3-7-sonnet-20250219",
    # "o3-pro-2025-06-10",
    #"grok-4",
    "kimi-k2",
    "glm-4.5",
    "qwen3-235b-thinking",
    "gpt-oss-20b",
    "gpt-oss-120b"
    ]

    titles = titles_A + titles_BCD + titles_E

    if size == 1:
        save_dir = "Responses_small"
    elif size == 2:
        save_dir = "Responses_med"
        titles = [s + "_med" for s in titles]
    else:
        save_dir = "Responses_large"
        titles = [s + "_large" for s in titles]
    
    save_all_results(titles, models, save_dir)
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)

    return


#ask_batch(1, 0, 10)
#print(status())
#save_batch(1, 10)
#cancel()





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------ Ask as a function of noise or resolution ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_single_prompt(question_category, resolution, noise_ratio):

    if question_category == "A1":
        Q, L = ask_A1(int(resolution), 0, noise_ratio)
    elif question_category == "B1_dispersion":
        Q, L = ask_B1_dispersion(int(resolution), 0, noise_ratio)
    elif question_category == "B1_vF":
        Q, L = ask_B1_vF(int(resolution), 0, noise_ratio)
    elif question_category == "B2_dispersion":
        Q, L = ask_B2_dispersion(int(resolution), 0, noise_ratio)
    elif question_category == "B2_vF":
        Q, L = ask_B2_vF(int(resolution), 0, noise_ratio)
    elif question_category == "B3_dispersion":
        Q, L = ask_B3_dispersion(int(resolution), 0, noise_ratio)
    elif question_category == "B3_vF":
        Q, L = ask_B3_vF(int(resolution), 0, noise_ratio)
    elif question_category == "B4_dispersion":
        Q, L = ask_B4_dispersion(int(resolution), 0, noise_ratio)
    elif question_category == "B4_bbE":
        Q, L = ask_B4_bbE(int(resolution), 0, noise_ratio)
    elif question_category == "B5":
        Q, L = ask_B5(int(resolution), 0, noise_ratio)
    elif question_category == "B6":
        Q, L = ask_B6(int(resolution), 0, noise_ratio)
    elif question_category == "C1":
        Q, L = ask_C1(int(resolution), 0, noise_ratio)
    elif question_category == "C2":
        Q, L = ask_C2(int(resolution), 0, noise_ratio)
    elif question_category == "C3":
        Q, L = ask_C3(int(resolution), 0, noise_ratio)
    elif question_category == "C4":
        Q, L = ask_C4(int(resolution), 0, noise_ratio)
    elif question_category == "C5":
        Q, L = ask_C5(int(resolution), 0, noise_ratio)
    elif question_category == "D1":
        coupling_lambda = 1
        Q, L = ask_D1(int(resolution), 0, noise_ratio, coupling_lambda)
    elif question_category == "D2":
        Q, L = ask_D2(int(resolution), 0, noise_ratio)
    elif question_category == "D3":
        Q, L = ask_D3(int(resolution), 0, noise_ratio)
    elif question_category == "E1":
        Q, L = ask_E1(int(resolution), 0, noise_ratio)
    elif question_category == "E2":
        Q, L = ask_E2(int(resolution), 0, noise_ratio)
    elif question_category == "E3":
        Q, L = ask_E3(int(resolution), 0, noise_ratio)
    else:
        Q, L = ask_E4(int(resolution), 0, noise_ratio)

    noise_int = round(100*noise_ratio)
    question = Q
    length = L
    title = question_category + f"_r{int(resolution)}_n{noise_int}"

    return question, length, title



def ask_resolution_iter(question_category, resolution_array, noise_ratio, attempts):

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
     "gemini-2.5-pro-preview-03-25"
    # "claude-3-7-sonnet-20250219"
    ]

    save_dir = "Responses_single"
    prompts = []; num_elems = []; titles = []

    for i in range(len(resolution_array)):
        question, length, title = get_single_prompt(question_category, int(resolution_array[i]), noise_ratio)
        prompts.append(question)
        num_elems.append(length)
        titles.append(title)

    print("Response lengths =", num_elems)
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)
    run(models, prompts, titles, num_elems, num_repeats = attempts_left)
    return



def ask_noise_iter(question_category, resolution, noise_ratio_array, attempts):

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
     "gemini-2.5-pro-preview-03-25"
    # "claude-3-7-sonnet-20250219"
    ]

    save_dir = "Responses_single"
    prompts = []; num_elems = []; titles = []

    for i in range(len(noise_ratio_array)):
        question, length, title = get_single_prompt(question_category, resolution, noise_ratio_array[i])
        prompts.append(question)
        num_elems.append(length)
        titles.append(title)

    print("Response lengths =", num_elems)
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)
    run(models, prompts, titles, num_elems, num_repeats = attempts_left)
    return



def save_resolution_iter(question_category, resolution_array, noise_ratio, attempts):

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
     "gemini-2.5-pro-preview-03-25"
    # "claude-3-7-sonnet-20250219"
    ]

    noise_int = round(100*noise_ratio)
    save_dir = "Responses_single"
    titles = []

    for i in range(len(resolution_array)):
        title = question_category + f"_r{int(resolution_array[i])}_n{noise_int}"
        titles.append(title)

    save_all_results(titles, models, save_dir)
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)
    return



def save_noise_iter(question_category, resolution, noise_ratio_array, attempts):

    models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner"
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
     "gemini-2.5-pro-preview-03-25"
    # "claude-3-7-sonnet-20250219"
    ]

    save_dir = "Responses_single"
    titles = []

    for i in range(len(noise_ratio_array)):
        noise_int = round(100*noise_ratio_array[i])
        title = question_category + f"_r{int(resolution)}_n{noise_int}"
        titles.append(title)

    save_all_results(titles, models, save_dir)
    attempts_left = get_attempts_left(models, titles, save_dir, attempts)
    print("Attempts left =", attempts_left)
    return



#resolution_array_o3 = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200])
#ask_resolution_iter("B1_dispersion", resolution_array_o3, 0, 30)
#resolution_array_Gemini = np.array([225, 250, 275, 300])
#resolution_array_total = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200, 225, 250, 275, 300])
#resolution_array_short = np.array([30])

#resolution_array_A1 = np.array([25, 30, 40, 50, 63, 75, 88, 100, 125, 150, 175, 200, 225, 250])
#ask_resolution_iter("A1", resolution_array_A1, 0, 100)
#ask_resolution_iter("B1_dispersion", resolution_array_total, 0, 100)

#print(status())

#save_resolution_iter("A1", resolution_array_A1, 0, 30)
#save_resolution_iter("B1_dispersion", resolution_array_total, 0, 30)

#save_resolution_iter("B1_dispersion", resolution_array_short, 0, 1)
#save_resolution_iter("B1_dispersion", resolution_array_o3, 0, 30)
#save_resolution_iter("B1_dispersion", resolution_array_Gemini, 0, 30)

#--------------------------------------------------------------------------------

#noise_array_total = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
#ask_noise_iter("A1", 100, noise_array_total, 100)
#ask_noise_iter("B1_dispersion", 100, noise_array_total, 100)
#ask_noise_iter("B2_dispersion", 100, noise_array_total, 100)
#ask_noise_iter("D1", 100, noise_array_total, 100)

#print(status())

#save_noise_iter("A1", 100, noise_array_total, 100)
#save_noise_iter("B1_dispersion", 100, noise_array_total, 100)
#save_noise_iter("B2_dispersion", 100, noise_array_total, 100)
#save_noise_iter("D1", 100, noise_array_total, 100)

#cancel()