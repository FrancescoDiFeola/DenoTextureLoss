import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from matplotlib.patches import ConnectionPatch
import numpy as np
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Latex preamble
# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#########################################################################

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Missing modalities
labels = ['Theoretically Robust', 'Not Robust', 'Robust']
sizes = [7, 84, 9]  # Percentages
# colors
colors = ['#66b3ff', '#ff9999', '#99ff99']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct=lambda p: "{:.1f}\%".format(p * sum(sizes) / 100, p), startangle=90, textprops={'family': 'serif', 'size': 20, 'weight': 'bold'}, colors=colors)
# draw circle
#centre_circle = plt.Circle((0, 0), 0.70, fc='white')

# fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Explainability
"""labels = ['Unimodal XAI', 'Multimodal XAI', "No XAI"]
labels_1 = ['XAI', "No XAI"]
sizes = [19, 8, 34]  # Percentages
sizes_1 = [20, 34]
# colors
colors = ['#99ff99', '#66b3ff', '#ff9999']
colors_1 = ['#99ff99', '#ff9999']

fig1, ax1 = plt.subplots()
ax1.pie(sizes_1, labels=labels_1, colors=colors_1, autopct=lambda p: "{:.0f}\n ({:.1f}\%)".format(p * sum(sizes_1) / 100, p), startangle=90, textprops={'family': 'serif', 'size': 14, 'weight': 'bold'})
# draw circle
#centre_circle = plt.Circle((0, 0), 0.70, fc='white')

# fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')

plt.tight_layout()
plt.show()"""

# Radar chart
# Data for the first radar chart
"""labels1 = np.array(['Imaging', 'Tabular', 'Text', 'Time series'])  # Categories/Labels
values1 = np.array([21, 12, 1, 3])  # Corresponding values for each category

# Data for the second radar chart
labels2 = np.array(['Imaging', 'Tabular', 'Text', 'Time series'])  # Categories/Labels
values2 = np.array([7, 8, 0, 2])  # Corresponding values for each category

# Number of variables/categories
num_vars = len(labels1)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Make the plot close to a circle
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

# First radar chart
# ax.fill(angles, values1, color='green', alpha=0.25, label='Unimodal XAI')
ax.plot(angles, values1, color='green', linewidth=2, label='Unimodal XAI')

# Second radar chart
# ax.fill(angles, values2, color='blue', alpha=0.25, label='Multimodal XAI')
ax.plot(angles, values2, color='blue', linewidth=2, label='Multimodal XAI')


# Set the number of grid lines equal to the number of categories
ax.set_yticks(np.arange(max(values1) + 1))
ax.set_yticklabels([str( ) for i in range(max(values1) + 1)], fontsize=8, fontweight='bold')

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels1, fontsize=20)

# Legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
# Display
plt.show()"""

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Disease areas
#Pie chart
"""labels = ['Oncology', 'Mental health', 'Pneumology', 'Others']
sizes = [20, 14, 9, 12]  # Percentages
# colors
# colors = ['#22fd33', '#66b3ff', '#99ff99', '#ff9999']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct=lambda p: "{:.0f}\n ({:.1f}\%)".format(p * sum(sizes) / 100, p), startangle=90, textprops={'family': 'serif', 'size': 14, 'weight': 'bold'})
# draw circle
#centre_circle = plt.Circle((0, 0), 0.70, fc='white')

# fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')

plt.tight_layout()
plt.show()"""


# Radar chart Data for the first radar chart
"""# labels1 = np.array(['CT', 'MRI', 'PET', 'X-ray', 'MFI', 'US', 'WSI', "Notes", "clinical", 'Genomics', 'chemical', 'Electrical', 'Physical', 'Speech', 'Video', 'eHR', 'OCT'])  # Categories/Labels
labels1 = np.array(['CT', 'MRI', 'PET', 'X-ray', 'MFI', 'US', 'WSI', "Q-A", "clinical", 'Genomics', 'chemical', "Physio", 'Speech', 'Video', 'eHR', 'OCT', "2D Fundus", "Skin lesion"])  # Categories/Labels

# values1 = np.array([8, 4, 2, 0, 0, 1, 2, 0, 7, 8, 2, 0, 0, 0, 0, 0, 0])  # Corresponding values for each category
values1 = np.array([7, 8, 2, 0, 0, 4, 1, 1, 8, 11, 7, 0, 0, 0, 0, 0, 0, 0])

# Data for the second radar chart
# labels2 = np.array(['CT', 'MRI', 'PET', 'X-ray', 'MFI', 'US', "Notes", "clinical", 'Genomics', 'chemical', 'WSI', 'Electrical', 'Physical', 'Speech', 'Video', 'eHR', 'OCT'])  # Categories/Labels
# values2 = np.array([0, 7, 0, 0, 0, 0, 0, 0, 3, 4, 0, 6, 2, 2, 1, 1, 0])  # Corresponding values for each category
values2 = np.array([0, 11, 0, 0, 0, 0, 0, 0, 4, 6, 0, 11, 1, 1, 0, 0, 0, 0])  # Corresponding values for each category

# Data for the third radar chart
# labels3 = np.array(['CT', 'MRI', 'PET', 'X-ray', 'MFI', 'US', "Notes", "clinical", 'Genomics', 'chemical', 'WSI', 'Physio', 'Speech', 'Video', 'eHR', 'OCT'])  # Categories/Labels
# values3 = np.array([3, 0, 0, 4, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 4, 0])  # Corresponding values for each category
values3 = np.array([3, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 2, 1, 0, 3, 0, 0, 0])


# Data for the fourth radar chart
# labels4 = np.array(['CT', 'MRI', 'PET', 'X-ray', 'MFI', 'US', "Notes", "clinical", 'Genomics', 'chemical', 'Whole slide images', 'Physiological signals', 'Speech', 'Video', 'eHR', 'OCT'])  # Categories/Labels
# values4 = np.array([0, 1, 1, 1, 1, 1, 0, 1, 4, 0, 0, 3, 1, 1, 2, 4, 1])  # Corresponding values for each category
values4 = np.array([0, 0, 0, 3, 1, 1, 0, 1, 8, 0, 0, 5, 0, 2, 1, 1, 1, 0])

# Number of variables/categories
num_vars = len(labels1)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Make the plot close to a circle
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
values4 = np.concatenate((values4, [values4[0]]))
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

# First radar chart
# ax.fill(angles, values1, color='blue', alpha=0.25, label='Oncology')
ax.plot(angles, values1, color='blue', label='Oncology', linewidth=2)

# Second radar chart
# ax.fill(angles, values2, color='orange', alpha=0.25, label='Mental Health')
ax.plot(angles, values2, color='orange', label='Mental Health', linewidth=2)

# Third radar chart
#ax.fill(angles, values3, color='green', alpha=0.25, label='Pneumology')
ax.plot(angles, values3, color='green', label='Pneumology',  linewidth=2)

# Fourth radar chart
# ax.fill(angles, values4, color='red', alpha=0.25, label='Others')
ax.plot(angles, values4, color='red', label='Others', linewidth=2)
# Adjusting radial positions of labels

# Set the number of grid lines equal to the number of categories
ax.set_yticks(np.arange(max(values1) + 1))
ax.set_yticklabels([str( ) for i in range(max(values1) + 1)], fontsize=8, fontweight='bold')

# Labels
# ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels1, fontsize=20)

# Legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Set the number of gridlines to match the number of elements
# ax.set_yticks(np.linspace(0, 13, 13))  # Adjust the range according to your data

#ax.grid(False)
# Display
plt.show()"""


# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Experimental configuration
# Avg(M) vs Avg(U)
"""categories = [r'$\bar{M}>\bar{U}$', r'$\bar{M}=\bar{U}$']
values_no_sigma = [27, 0]
values_sigmaM_lower_sigmaU = [4, 0]
values_sigmaM_higher_sigmaU = [2, 0]
values_sigmaM_equal_sigmaU = [5, 1]


# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.barh(categories, values_no_sigma, label=r'w/o $\sigma$')
bar2 = ax.barh(categories, values_sigmaM_lower_sigmaU, left=values_no_sigma, label=r'$\sigma(M)<\sigma(U)$')
bar3 = ax.barh(categories, values_sigmaM_higher_sigmaU, left=[31, 0], label=r'$\sigma(M)>\sigma(U)$')
bar4 = ax.barh(categories, values_sigmaM_equal_sigmaU, left=[33, 0], label=r'$\sigma(M)=\sigma(U)$')
ax.set_xlabel('Number of works')
ax.legend()
plt.grid(False)
plt.tight_layout()
plt.show()"""

# Joint vs Others
"""categories = [r'$\bar{JF}>(\bar{EF} \land \bar{LF})$', r'$\bar{EF}>(\bar{JF} \land \bar{LF})$', r'$\bar{LF}>(\bar{JF} \land \bar{EF})$']
values_no_sigma = [16, 1, 1]
values_sigmaJ_lower_sigmaE_sigma_L = [2, 0, 0]
values_sigmaJ_equal_sigmaE_sigmaL = [4, 1, 1]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.barh(categories, values_no_sigma, label=r'w/o $\sigma$')
bar2 = ax.barh(categories, values_sigmaJ_lower_sigmaE_sigma_L, left=values_no_sigma, label=r'$\sigma_{JF}<(\sigma_{EF} \land \sigma_{LF})$')
bar3 = ax.barh(categories, values_sigmaJ_equal_sigmaE_sigmaL, left=[18, 1, 1], label=r'$\sigma_{JF} = \sigma_{EF} = \sigma_{LF}$')
ax.set_xlabel('Number of works')
ax.legend()
plt.grid(False)
plt.tight_layout()
plt.show()"""


# Radar chart
# Data for the first radar chart
"""labels1 = np.array(['A', r'$A \wedge B$', r'$A \wedge C$', r'$A \wedge E$', r'$A \wedge B \wedge C$', r'$A \wedge B \wedge C \wedge D$', r'$A \wedge B \wedge C \wedge E$', r'$A \wedge B \wedge C \wedge D \wedge E$', 'B', r'$B \wedge C$', r'$B \wedge C \wedge E$', 'C', r'$C \wedge E$', 'E'])  # Categories/Labels
values1 = np.array([16, 6, 2, 2, 7, 2, 1, 3, 7, 1, 1, 2, 1, 1])  # Corresponding values for each category


# Number of variables/categories
num_vars = len(labels1)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Make the plot close to a circle
values1 = np.concatenate((values1, [values1[0]]))
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

# First radar chart
ax.fill(angles, values1, color='green', alpha=0.25, label='Unimodal XAI')
ax.plot(angles, values1, color='green', linewidth=2)


# Labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels1, fontsize=15)

# Display
plt.show()"""

# Disease based analysis
"""categories = ['Oncology', 'Mental Health', 'Pneumology', 'Other diseases']
values_CT = [8, 0, 3, 0]
values_MRI = [4, 7, 0, 1]
values_PET = [2, 0, 1, 0]
values_xRay = [0, 0, 1, 4]
values_MFI = [0, 0, 1, 0]
values_US = [1, 0, 1, 0]
values_notes = [0, 0, 1, 1]
values_clinical = [7, 3, 5, 3]
values_genomics = [8, 4, 0, 0]
values_chemical = [2, 0, 0, 0]
values_wsi = [2, 0, 0, 0]
values_pysio = [0, 6, 4, 0]
values_speech = [0, 2, 1, 0]
values_video = [0, 1, 2, 0]
values_ehr = [0, 1, 1, 4]
values_oct = [0, 0, 1, 0]

values_prognosis = [11, 4, 0, 0]
values_detection = [8, 0, 2, 1]
values_drug = [2, 0, 0, 0]
values_qa = [1, 0, 0, 0]




# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

bar1 = ax.barh(categories, values_CT, color='#1f77b4',  label='CT')
bar2 = ax.barh(categories, values_MRI, left=values_CT, color='#ff7f0e', label='MRI')
bar3 = ax.barh(categories, values_PET, left=[x + y for x, y in zip(values_MRI, values_CT)], color='#2ca02c', label='PET')
bar4 = ax.barh(categories, values_xRay, left=[x + y + z for x, y, z in zip(values_MRI, values_CT, values_PET)], color='#d62728', label='X-Ray')
bar5 = ax.barh(categories, values_MFI, left=[x + y + z + t for x, y, z, t in zip(values_MRI, values_CT, values_PET, values_xRay)], color='#9467bd', label='MFI')
bar6 = ax.barh(categories, values_US, left=[x + y + z + t + a for x, y, z, t, a in zip(values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#8c564b', label='US')
bar7 = ax.barh(categories, values_notes, left=[x + y + z + t + a + b for x, y, z, t, a, b in zip(values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#e377c2', label='Notes')
bar8 = ax.barh(categories, values_clinical, left=[x + y + z + t + a + b + c for x, y, z, t, a, b, c in zip(values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#7f7f7f', label='Clinical')
bar9 = ax.barh(categories, values_genomics, left=[x + y + z + t + a + b + c + d for x, y, z, t, a, b, c, d in zip(values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#bcbd22', label='Genomics')
bar10 = ax.barh(categories, values_notes, left=[x + y + z + t + a + b + c + d + e for x, y, z, t, a, b, c, d, e in zip(values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#17becf', label='Chemical')
bar11 = ax.barh(categories, values_wsi, left=[x + y + z + t + a + b + c + d + e + f for x, y, z, t, a, b, c, d, e, f in zip(values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#aec7e8', label='WSI')
bar12 = ax.barh(categories, values_pysio, left=[x + y + z + t + a + b + c + d + e + f + g for x, y, z, t, a, b, c, d, e, f, g in zip(values_wsi, values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#ffbb78', label='Physiological signals')
bar13 = ax.barh(categories, values_speech, left=[x + y + z + t + a + b + c + d + e + f + g + h for x, y, z, t, a, b, c, d, e, f, g, h in zip(values_pysio, values_wsi, values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#98df8a', label='Speech')
bar14 = ax.barh(categories, values_video, left=[x + y + z + t + a + b + c + d + e + f + g + h + i for x, y, z, t, a, b, c, d, e, f, g, h, i in zip(values_speech, values_pysio, values_wsi, values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#ff9896', label='Video')
bar15 = ax.barh(categories, values_ehr, left=[x + y + z + t + a + b + c + d + e + f + g + h + i + j for x, y, z, t, a, b, c, d, e, f, g, h, i, j in zip(values_video, values_speech, values_pysio, values_wsi, values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color='#c5b0d5', label='eHR')
bar16 = ax.barh(categories, values_oct, left=[x + y + z + t + a + b + c + d + e + f + g + h + i + j + k for x, y, z, t, a, b, c, d, e, f, g, h, i, j, k in zip(values_ehr, values_video, values_speech, values_pysio, values_wsi, values_notes, values_genomics, values_clinical, values_notes, values_US, values_MFI, values_MRI, values_CT, values_PET, values_xRay)], color= '#ff8c00', label='OCT')
ax.set_xlabel('Number of works')

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#ff8c00'
]
# Create custom legend handles
handles = [plt.Rectangle((0,0),1,1, color=colors[i], ec="k", lw=0.5) for i in range(len(colors))]
labels = ['CT', 'MRI', 'PET', 'X-Ray', 'MFI', 'US', 'Notes', 'Clinical', 'Genomics', 'Chemical', 'WSI', 'Physiological signals', 'Speech', 'Video', 'eHR', 'OCT']

# Splitting the legend
leg1 = plt.legend(handles[:4], labels[:4], loc='upper right', bbox_to_anchor=(0.7, 1))
leg2 = plt.legend(handles[4:], labels[4:], loc='upper right', bbox_to_anchor=(1, 1))

# Add both legends to the same axis
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.grid(False)
plt.tight_layout()
# ax.legend()

plt.show()"""

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Tasks based analysis
"""categories = ['Classification', 'Regression', 'Segmentation', 'Object detection']
values_diagnosis = [27, 0, 0, 0]
values_prognosis = [11, 4, 0, 0]
values_detection = [8, 0, 2, 1]
values_drug = [2, 0, 0, 0]
values_qa = [1, 0, 0, 0]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.barh(categories, values_diagnosis, label='Diagnosis')
bar2 = ax.barh(categories, values_prognosis, left=values_diagnosis, label='Prognosis')
bar3 = ax.barh(categories, values_detection, left=[38, 4, 0, 0], label='Detection')
bar4 = ax.barh(categories, values_drug, left=[46, 4, 2, 1], label='Drug discovery')
bar5 = ax.barh(categories, values_qa, left=[47, 4, 2, 1], label='Question answering')

ax.set_xlabel('Number of works')
ax.legend()
plt.grid(False)
plt.tight_layout()
plt.show()"""
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# Medical task-modalities

"""categories = ['Diagnosis', 'Prognosis', 'Detection', 'Object detection']
values_diagnosis = [27, 0, 0, 0]
values_prognosis = [11, 4, 0, 0]
values_detection = [8, 0, 2, 1]
values_drug = [2, 0, 0, 0]
values_qa = [1, 0, 0, 0]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.barh(categories, values_diagnosis, label='Diagnosis')
bar2 = ax.barh(categories, values_prognosis, left=values_diagnosis, label='Prognosis')
bar3 = ax.barh(categories, values_detection, left=[38, 4, 0, 0], label='Detection')
bar4 = ax.barh(categories, values_drug, left=[46, 4, 2, 1], label='Drug discovery')
bar5 = ax.barh(categories, values_qa, left=[47, 4, 2, 1], label='Question answering')

ax.set_xlabel('Number of works')
ax.legend()
plt.grid(False)
plt.tight_layout()
plt.show()"""


