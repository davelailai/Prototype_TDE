# FFA [0.4958974358974359, 0.6324237356846052, 0.5001225251063018, 0.7227500158338083, 0.7508091396550891, 0.7638668166682246,
# 0.5821201170194459, 0.6697957839262187, 0.6234145478142883, 0.768972069162075, 0.8130653913659566, 0.8420776935003982,
# 0.5, 0.5, 0.5, 0.5, 0.5, 0.8236600595063487,
# 0.6149750473240405, 0.6356478666261274, 0.62875119688877, 0.7633542339603522, 0.8186480524290292, 0.83724562325325,
# 0.630917225950783, 0.6619476537954798, 0.6552801513865758, 0.7634302362404205, 0.8222678453536866, 0.8387558466760578,
# 0.6464257442780934, 0.6336272423228945, 0.6568525569174498, 0.7434986382924822, 0.8017358259665396, 0.8321750118464698,
# 0.6283978661159869, 0.6808237052802271, 0.6406089043986514, 0.7631610614985116, 0.8209506787111733, 0.8457041638326473
# ]
# ['Resnet50','Asyloss','GCN','Q2L','Q2L causal','ML Decoder','ML causal']

# OD[0.5482573726541555, 0.7401041666666667, 0.912565445026178, 0.7976867045545506, 0.7310996563573884, 0.959417273673257, 0.5335294117647058,
#     0.7802611302273542, 0.7911458333333332, 0.966259453170448, 0.8641372501934416, 0.6029209621993127, 0.9429947437231517, 0.6501960784313725,
#     0.8034240358024951, 0.7026041666666667, 0.9789412449098314, 0.8121881587022065, 0.6561855670103093, 0.9772939512793831, 0.6937254901960784,
#     0.7813588482404847, 0.8057291666666667, 0.9117510180337406, 0.835040956268844, 0.6154639175257732, 0.9674084153792791, 0.6613725490196078,
#     0.810390323193515, 0.7973958333333333, 0.9732984293193717, 0.8596413991835428, 0.6878006872852234, 0.9249713172710052, 0.6650980392156863,
#     0.7827679381900319, 0.6296875, 0.9503199534613146, 0.835040956268844, 0.7374570446735395, 0.9443688465540703, 0.6703921568627451,
#     0.7991228810875853, 0.7791666666666668, 0.9625945317044795, 0.8975826462819179, 0.7214776632302407, 0.9398729955441714, 0.6758823529411765

# ]
from curses import raw
from tkinter.ttk import Style
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.ticker import MaxNLocator

import pandas as pd
import matplotlib.pyplot as plt



# Given data
result_dist = {'Dataset': ['ODIR-5K (on-site)'] * 56 +['ODIR-5K (off-site)'] * 56 +['Chexpert']*101+ ['LID-FFA'] * 48+ ['Endo'] * 32,
            #    'Model': ['DenseNet121'] * 7 + ['Asy'] * 7 + ['GCN'] * 7 + ['dyGCN']*7 + ['Q2L'] * 7 + ['ML'] * 7 + ['MCAR'] * 7 + ['Our'] * 7 +
            #             ['DenseNet121'] * 7 + ['Asy'] * 7 + ['GCN'] * 7 + ['dyGCN']*7 + ['Q2L'] * 7 + ['ML'] * 7 + ['MCAR'] * 7 + ['Our'] * 7 +
            #             ['DenseNet121'] * 6 + ['Asy'] * 6 + ['GCN'] * 6 + ['dyGCN']*6 + ['Q2L'] * 6 + ['ML'] * 6 + ['MCAR'] * 6 + ['Our'] * 6,
                'Model': ['Asy'] * 7 + ['GCN'] * 7 + ['dyGCN']*7 + ['Q2L'] * 7 + ['ML'] * 7 + ['MCAR'] * 7 +['Q2L_causal']*7+ ['Our'] * 7 +
                        ['Asy'] * 7 + ['GCN'] * 7 + ['dyGCN']*7 + ['Q2L'] * 7 + ['ML'] * 7 + ['MCAR'] * 7 +['Q2L_causal']*7+ ['Our'] * 7 +
                        ['Asy'] * 13 + ['GCN'] * 13 + ['dyGCN']*13 + ['Q2L'] * 13 + ['ML'] * 13 + ['MCAR'] * 13 +['Q2L_causal']*13+ ['Our'] * 13 +
                        ['Asy'] * 6 + ['GCN'] * 6 + ['dyGCN']*6 + ['Q2L'] * 6 + ['ML'] * 6 + ['MCAR'] * 6 +['Q2L_causal']*6+ ['Our'] * 6 +
                        ['Asy'] * 4 + ['GCN'] * 4 + ['dyGCN']*4 + ['Q2L'] * 4 + ['ML'] * 4+ ['MCAR'] * 4 +['Q2L_causal']*4+ ['Our'] * 4,
               'AUC': [
                        #  0.7804625822368421, 0.9143302180685359, 0.9740516645945573, 0.8461894017736831, 0.7389707281793613, 0.9628090366581415, 0.5297913232278557,
                         0.8166920230263157, 0.8192795599337656, 0.9787318690426854, 0.8574237723668955, 0.8069734688439725, 0.9645369017172086, 0.6070089662659321,
                         0.8124917763157895, 0.8755578007914456, 0.9648570244508909, 0.8742753282567141, 0.6499371931026607, 0.96622290220436, 0.6658051545977243,
                         0.8298506030701753, 0.9096082063371783, 0.9711341345489709, 0.9111043213938766, 0.8346846332457842, 0.9678099500669833, 0.6675935828877004,
                         0.830670230263158, 0.9121060312649097, 0.9764000552562508, 0.902698129171759, 0.8170035400251228, 0.9745311167945439, 0.6489775240239637,
                         0.8173780153508773, 0.7287439308467347, 0.9766045033844454, 0.8592804988875309, 0.7922423965589432, 0.968091584459871, 0.577797434763379,
                         0.8505619517543859, 0.8542420925598496, 0.9632877469263711, 0.8980915671711949, 0.8192112976285638, 0.9646815247838265, 0.6972248803827751,
                         0.837140899122807, 0.9088644719485841, 0.9736759220886863, 0.9095923035943719, 0.7927182063872711, 0.9727499695530386, 0.6975722729283101,
                         0.8334580592105263, 0.9061912379669388, 0.9803398259428098, 0.9257466077528125, 0.8371207795668227, 0.9770734380708805, 0.707255840135097,
                         
                      
                         0.8708439762724028, 0.7565364583333334, 0.9873647469458988, 0.927772886149577, 0.733298969072165, 0.9936497772085701, 0.6484627450980391,
                         0.8569061239999156, 0.8656510416666667, 0.9951134380453752, 0.9503455268283573, 0.7432646048109965, 0.9761599829237706, 0.7115058823529412,
                         0.8832671887863882, 0.8446875, 0.9924607329842933, 0.9367379065610075, 0.761512027491409, 0.9938632300755089, 0.6535686274509804,
                         0.8916425661269552, 0.8595572916666667, 0.9904363001745201, 0.9669414872328503, 0.7848453608247423, 0.9965580725206115, 0.6930823529411764,
                         0.8810137003652022, 0.7201822916666667, 0.9926701570680628, 0.9203287174150857, 0.7246735395189003, 0.9949571760185704, 0.6399294117647059,
                         0.9042241033543729, 0.8422395833333333, 0.9926701570680628, 0.9398596547399877, 0.755085910652921, 0.9937565036420396, 0.6966745098039215,
                         0.8924236347132212, 0.8473437500000001, 0.9891564863292612, 0.9604845380079512, 0.807938144329897, 0.9943168174177539, 0.7259058823529412,
                         0.8876211184057756, 0.8480729166666667, 0.9921582315299593, 0.967368392966728, 0.8214089347079039, 0.9964513460871421, 0.7260078431372547,

                         0.8373299473970615, 0.8583019414662416, 0.9128041088484411, 0.7779598077763216, 0.9244274038946626, 0.9040848566914917, 0.9384010484927916, 0.8706718642513185, 0.8828267477203648, 0.9489811435523114, 0.9477272727272728, 0.6915911379657603, 0.9431988848419443,
                         0.8544667150371849, 0.8584410315850478, 0.9113488916921968, 0.7980559196155526, 0.9207950761779841, 0.8949221394719025, 0.8966797728265619, 0.8767140564090805, 0.9086626139817628, 0.9460158150851582, 0.9234848484848486, 0.6563444108761329, 0.9456090651558073,
                         0.8509160166878287, 0.855137641263402, 0.9103351955307264, 0.846439493228484, 0.9163959237211179, 0.8810200857594223, 0.882699868938401, 0.8722770006879156, 0.8872340425531915, 0.9426399026763991, 0.9397727272727273, 0.784994964753273, 0.9324340123206978,
                         0.8456375838926175, 0.8648623587365981, 0.9158136601189404, 0.8344255133245959, 0.9216022601150238, 0.8838185511171294, 0.8851026649191787, 0.8717495987158909, 0.9082066869300913, 0.9422141119221411, 0.9634469696969696, 0.7590634441087614, 0.9305993974549216,
                         0.845456194449483, 0.8535844682700666, 0.9178140205442422, 0.8040629095674967, 0.9192008879023308, 0.8829158203565787, 0.8966797728265619, 0.8717152029351067, 0.8734042553191489, 0.9478862530413624, 0.9651515151515151, 0.7696374622356495, 0.9368586717028645,
                         0.8404861237076002, 0.8588698927847, 0.9063344746801226, 0.8211009174311926, 0.9067298960750683, 0.9018731663281425, 0.8975535168195719, 0.8797523503783535, 0.8472644376899696, 0.9420620437956204, 0.9535984848484849, 0.7817220543806646, 0.9289806196321777,
                         
                         0.8514647197533104, 0.8629498696030136, 0.9225175707334655, 0.8014416775884665, 0.9153667642013924, 0.899616339426766, 0.9398208824814329, 0.8733318046319651, 0.9575987841945289, 0.9370894160583942, 0.950189393939394, 0.8081570996978852, 0.9414452088673052,
                         
                        
                         0.7807210678409248, 0.7500332000903042, 0.8177569523995223, 0.8779016557117896, 0.9170018867860151, 0.9232993342437692,
                         0.7725964405302509, 0.6975305772831701, 0.7862853103902119, 0.8633596805058656, 0.8966583757263101, 0.9326295237282348,
                         0.7847690472914086, 0.7565337777718755, 0.807182115089327, 0.8726516349113903, 0.9188101988802113, 0.9344006060088768,
                         0.8106107518003761, 0.769322452557071, 0.7967382826781058, 0.8844845661036693, 0.9164184502424162, 0.9296154831000341,
                        #  0.8363549837163434, 0.7534793694638849, 0.791790562772673, 0.8722139945086945, 0.9045211214264511, 0.9295585808580858,
                         0.8123595247924408, 0.7063385612408866, 0.8026822491408515, 0.8627173641733922, 0.9150127434823997, 0.9204115454648912,
                         0.8053644328241825, 0.7492397179320328, 0.8000865241658339, 0.8888526499708793, 0.9086392963277617, 0.9355404290429044,
                         0.8299848630796752, 0.7730126425943878, 0.8136348680201807, 0.8818504035277477, 0.9202534366459564, 0.935824940252646,
                         0.8640027980367873, 0.7910170515663801, 0.8479551049257842, 0.8778633829769531, 0.9275515113053625, 0.9333656964834415,
                         #  0.35100454107609746, 0.6656684506181857, 0.22808866899022645, 0.8518911723105084, 0.9239621823938641, 0.9054675941732105,

                         
                         0.7523774612484289, 0.8432414421997755, 0.8317611981936429, 0.9487744159325929,
                         0.7572685379136993, 0.829869879349046, 0.8306278660227014, 0.9716104940635772,
                         0.7669983242563888, 0.8462138748597081, 0.8519345108363992, 0.9638548448870164,
                         0.7448366149979053, 0.8513257575757576, 0.8540616881418582, 0.9685944082726924,
                         0.732069543359866, 0.8323162177328844, 0.7787735602322459, 0.9019532746074301,
                         0.7669983242563887, 0.8630488916947252, 0.8280822276079716, 0.9347950976637305,
                         0.7707896941767911, 0.8434694163860832, 0.8542186110578348, 0.9658655687476062,
                         0.7817134478424802, 0.8649252946127947, 0.8587693756211533, 0.984153581003447

                        #  0.7847690472914086, 0.7565337777718755, 0.807182115089327, 0.8726516349113903, 0.9188101988802113, 0.9344006060088768,
                        #  0.8106107518003761, 0.769322452557071, 0.7967382826781058, 0.8844845661036693, 0.9164184502424162, 0.9296154831000341,
                        # #  0.8363549837163434, 0.7534793694638849, 0.791790562772673, 0.8722139945086945, 0.9045211214264511, 0.9295585808580858,
                        #  0.8123595247924408, 0.7063385612408866, 0.8026822491408515, 0.8627173641733922, 0.9150127434823997, 0.9204115454648912,
                        #  0.8173420943993395, 0.7438430432530776, 0.8178513977918058, 0.8750544970463432, 0.9299483778075752, 0.9310220353931946,
                        #  0.8299848630796752, 0.7730126425943878, 0.8136348680201807, 0.8818504035277477, 0.9202534366459564, 0.935824940252646,
                        #  0.8640027980367873, 0.7910170515663801, 0.8479551049257842, 0.8778633829769531, 0.9275515113053625, 0.9333656964834415
                        ],
               'Disease': ['D', 'G', 'C', 'A', 'H', 'M', 'O'] * 8 + ['D', 'G', 'C', 'A', 'H', 'M', 'O'] * 8 + ['Enl', 'Card', 'LungOp', 'LungLe', 'Edema', 'Consol', 'Pneu', 'Ate','Ptx','PleEff', 'PleOth', 'Frac', 'SupDe']*8+ ['L', 'TP', 'ST', 'SD', 'NP', 'VA'] * 8+ ['U', 'E', 'P', 'T'] * 8}

# # Create DataFrame
# all_data = pd.DataFrame(result_dist)

# unique_datasets = all_data['Dataset'].unique()

# # Iterate over unique datasets and draw separate bar plots
# for dataset in unique_datasets:
#     data_subset = all_data[all_data['Dataset'] == dataset]
#     plt.figure(figsize=(12,6))
#     ax = sns.barplot(data=data_subset, x='Disease', y='AUC', hue='Model',palette='Paired')
#     plt.title(f'AUC Analysis for {dataset}', fontsize=22)
#     plt.xlabel('Disease', fontsize=22)
#     plt.ylabel('AUC', fontsize=22)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     ax.set_ylim([0.4, 1]) 
#     plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18,title_fontsize='22')
#     plt.tight_layout()
#     plt.savefig(f'work_dirs_MIL/AUC_analysis_{dataset}.png', dpi=400, bbox_inches='tight')

# # ### radar chart
# # # Create DataFrame
# datasets = list(set(result_dist['Dataset']))
all_data = pd.DataFrame(result_dist)

# 获取唯一的数据集
datasets = all_data['Dataset'].unique()
# Define function to plot radar chart
# def plot_radar_chart(dataset):
#     # Filter data for the given dataset
#     indices = [i for i, ds in enumerate(result_dist['Dataset']) if ds == dataset]
#     model_data = [result_dist['Model'][i] for i in indices]
#     auc_data = [result_dist['AUC'][i] for i in indices]

#     # Get unique models
#     unique_models = list(set(model_data))

#     # Initialize figure and axis
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

#     # Create a color palette
#     colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))

#     # Plot radar chart for each model
#     for i, model in enumerate(unique_models):
#         model_indices = [j for j, m in enumerate(model_data) if m == model]
#         auc_values = [auc_data[j] for j in model_indices]
        

#         # Wrap around to complete the circle
#         auc_values += auc_values[:1]
#         ax.plot(np.linspace(0, 2 * np.pi, len(auc_values)), auc_values, label=model, color=colors[i])

#     # Add legend
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

#     # Set title
#     ax.set_title(f'AUC Analysis for {dataset}')

#     ax.set_ylim([0.6, 1.0])

#     # Save plot
#     plt.savefig(f'work_dirs_MIL/AUC_analysis_{dataset}_radar_chart.png', bbox_inches='tight')

##### ---------------------drow radar seperately -----------------
# def plot_radar_chart(dataset):
#     # 筛选出特定数据集的数据
#     data_subset = all_data[all_data['Dataset'] == dataset]
    
#     # 获取疾病种类作为极坐标轴标签
#     diseases = data_subset['Disease'].unique()
#     num_diseases = len(diseases)
    
#     # 获取模型种类
#     models = data_subset['Model'].unique()
    
#     # 设定雷达图
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

#     # 设置颜色
#     colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
#     # 绘制每个模型的AUC值
#     for i, model in enumerate(models):
#         # 获取当前模型的AUC数据
#         model_data = data_subset[data_subset['Model'] == model]
        
#         # 按疾病顺序排列AUC
#         auc_values = [model_data[model_data['Disease'] == disease]['AUC'].values[0] for disease in diseases]
        
#         # 将第一个AUC值添加到最后以形成闭合曲线
#         auc_values += auc_values[:1]
        
#         # 设置角度
#         angles = np.linspace(0, 2 * np.pi, num_diseases, endpoint=False).tolist()
#         angles += angles[:1]  # 闭合
        
#         # 若是 "Our" 模型，设置更粗的线条、更醒目的填充色和标注值
#         if model == 'Our':
#             ax.plot(angles, auc_values, label=model, color='crimson', linewidth=2.5)  # 更粗的线条
#             ax.fill(angles, auc_values, color='crimson', alpha=0.2)  # 填充颜色
#             # 在每个轴上标出AUC值
#             for angle, auc in zip(angles[:-1], auc_values[:-1]):
#                 ax.text(angle, auc + 0.02, f'{auc:.2f}', color='crimson', ha='center', fontsize=10, fontweight='bold')
#         else:
#             # 其他模型，标准线条和颜色
#             ax.plot(angles, auc_values, label=model, color=colors[i], linewidth=1.5)


#     # 设置标签
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(diseases)
#     ax.set_ylim([0.4, 1.0])
#     ax.set_title(f'AUC Analysis for {dataset}')

#     # 添加图例
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

#     # 保存图像
#     plt.savefig(f'work_dirs_MIL/AUC_analysis_{dataset}_radar_chart_disease.png', bbox_inches='tight')
#     plt.show()
# # Plot radar chart for each dataset and save
# for dataset in datasets:
#     plot_radar_chart(dataset)

##### ---------------------drow radar in one -----------------
# 将数据转换为DataFrame
all_data = pd.DataFrame(result_dist)

# 获取唯一的数据集
datasets = all_data['Dataset'].unique()
selected_dataset = ['ODIR-5K (on-site)', 'LID-FFA','Endo','Chexpert']  # 修改为你想绘制的数据集名称
datasets = [ds for ds in datasets if ds in selected_dataset]
    

# 设置图像和子图
fig, axes = plt.subplots(1, len(datasets), figsize=(12 * len(datasets), 8), subplot_kw=dict(polar=True))

# title_height = 1.0
title_pad = 30  # 根据需要调整标题的高度偏移
title_y_position = 1.0  # 根据需要调整标题的垂直位置
# 创建颜色调色板
# colors = plt.cm.viridis(np.linspace(0, 1, len(all_data['Model'].unique())))
colors = sns.color_palette("husl", len(all_data['Model'].unique()))
# 遍历每个数据集，绘制雷达图
for ax, dataset in zip(axes, datasets):
    # 筛选出特定数据集的数据
    data_subset = all_data[all_data['Dataset'] == dataset]
    
    # 获取疾病种类作为极坐标轴标签
    diseases = data_subset['Disease'].unique()
    num_diseases = len(diseases)
    
    # 获取模型种类
    models = data_subset['Model'].unique()
    
    auc_values_all = [data_subset[data_subset['Model'] == model]['AUC'].values for model in models]
    auc_values_flat = np.concatenate(auc_values_all)
    min_auc = np.min(auc_values_flat)
    max_auc = np.max(auc_values_flat)
    # if min_auc<0.5:
    #     min_auc=0.5
    
    # 设置 Y 轴的动态范围（根据数据集的 AUC 值）
    ax.set_ylim([min_auc-0.05, max_auc+0.05]) 
    
    # 绘制每个模型的AUC值
    for i, model in enumerate(models):
        # 获取当前模型的AUC数据
        model_data = data_subset[data_subset['Model'] == model]
        
        # 确保按照疾病顺序排列AUC
        auc_values = [model_data[model_data['Disease'] == disease]['AUC'].values[0] for disease in diseases]
        
        # 将第一个AUC值添加到最后以形成闭合曲线
        auc_values += auc_values[:1]
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, num_diseases, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 若是 "Our" 模型，设置更粗的线条、更醒目的填充色和标注值
        if model == 'Our':
            ax.plot(angles, auc_values, label=model, color='crimson', linewidth=2.5)  # 更粗的线条
            ax.fill(angles, auc_values, color='crimson', alpha=0.3)  # 填充颜色
            # 在每个轴上标出AUC值
            for angle, auc in zip(angles[:-1], auc_values[:-1]):
                ax.text(angle, auc + 0.02, f'{auc:.2f}', color='crimson', ha='center', fontsize=25, fontweight='bold')
        else:
            # 其他模型，标准线条和颜色
            ax.plot(angles, auc_values, label=model, color=colors[i], linewidth=1.5)
            # ax.fill(angles, auc_values, color=colors[i], alpha=0.2) 

    # 设置子图标签
    ax.set_aspect('equal')
    ax.set_xticks(angles[:-1])
    # ax.set_yticks([])
  
    ax.set_xticklabels(diseases)
    # ax.set_ylim([0.7, 1.0])
    ax.tick_params(axis='y', labelsize=25) 
    ax.tick_params(axis='x', labelsize=25) 
    # ticks = [0.4, 0.8, 0.85, 0.9, 0.95, 1.0]
    # ax.set_yticks(ticks)  # 设置自定义的 Y 轴刻度
    
    # ax.yaxis.set_visible(True) 
    ax.set_yticklabels([]) 
    ax.set_title(f'AUC Analysis in {dataset}', fontsize=25, pad=title_pad, y=title_y_position)

# 创建全局图例
fig.legend(models, loc='upper right', bbox_to_anchor=(0.85, 1), title="Models", fontsize=25,title_fontsize=25)
# plt.subplots_adjust(wspace=0.5)  # 调整子图之间的水平间距
plt.grid(color='gray', linestyle='--', linewidth=0.5)  # 添加网格线

# 保存并显示图像
plt.tight_layout(rect=[0, 0, 0.8, 1])
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.05)
plt.savefig('work_dirs_BankCausal/ALL/AUC_analysis_combined_radar_chart_new.png', bbox_inches='tight')
plt.show()


# result_dist = {'Dataset': ['OIA-ODIR (on-site)'] * 7 + ['LID-FFA'] * 7,
#                'Model': ['Ours'] * 14,
#                'mAUC': [0.8814, 0.8626, 0.8617, 0.8743, 0.8882, 0.8716, 0.8686,
#                         0.8543, 0.8431, 0.8553, 0.8516, 0.8390, 0.8467, 0.8404],
#                'Channel': ['1', '5', '10', '20','50','100','196']*2}
# all_data = pd.DataFrame(result_dist)

# unique_Dataset = all_data['Dataset'].unique()
# # markers = {'Q2L causal': 'o', 'ML causal': '*'}
# for dataset in unique_Dataset:
#     plt.figure(figsize=(7, 3))
#     model_data = all_data[all_data['Dataset'] == dataset]
#     sns.lineplot(data=model_data, x='Channel', y='mAUC', hue='Model',palette='Set2')
#     plt.title(f'Channel Analysis in {dataset}', fontsize=16)
#     plt.xlabel('Top-K', fontsize=14)
#     plt.ylabel('mAUC', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,title_fontsize='12')
#     plt.tight_layout()
#     plt.savefig(f'work_dirs_MIL/Top_K_analysis_{dataset}.png', dpi=400, bbox_inches='tight')