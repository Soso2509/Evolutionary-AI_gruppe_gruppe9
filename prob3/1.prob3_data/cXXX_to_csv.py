import pandas as pd

# Define the data as a list of lists
c101 = [
[0, 40, 50, 0, 0, 1236, 0],
[1, 45, 68, 10, 912, 967, 90],
[2, 45, 70, 30, 825, 870, 90],
[3, 42, 66, 10, 65, 146, 90],
[4, 42, 68, 10, 727, 782, 90],
[5, 42, 65, 10, 15, 67, 90],
[6, 40, 69, 20, 621, 702, 90],
[7, 40, 66, 20, 170, 225, 90],
[8, 38, 68, 20, 255, 324, 90],
[9, 38, 70, 10, 534, 605, 90],
[10, 35, 66, 10, 357, 410, 90],
[11, 35, 69, 10, 448, 505, 90],
[12, 25, 85, 20, 652, 721, 90],
[13, 22, 75, 30, 30, 92, 90],
[14, 22, 85, 10, 567, 620, 90],
[15, 20, 80, 40, 384, 429, 90],
[16, 20, 85, 40, 475, 528, 90],
[17, 18, 75, 20, 99, 148, 90],
[18, 15, 75, 20, 179, 254, 90],
[19, 15, 80, 10, 278, 345, 90],
[20, 30, 50, 10, 10, 73, 90],
[21, 30, 52, 20, 914, 965, 90],
[22, 28, 52, 20, 812, 883, 90],
[23, 28, 55, 10, 732, 777, 90],
[24, 25, 50, 10, 65, 144, 90],
[25, 25, 52, 40, 169, 224, 90],
[26, 25, 55, 10, 622, 701, 90],
[27, 23, 52, 10, 261, 316, 90],
[28, 23, 55, 20, 546, 593, 90],
[29, 20, 50, 10, 358, 405, 90],
[30, 20, 55, 10, 449, 504, 90],
[31, 10, 35, 20, 200, 237, 90],
[32, 10, 40, 30, 31, 100, 90],
[33, 8, 40, 40, 87, 158, 90],
[34, 8, 45, 20, 751, 816, 90],
[35, 5, 35, 10, 283, 344, 90],
[36, 5, 45, 10, 665, 716, 90],
[37, 2, 40, 20, 383, 434, 90],
[38, 0, 40, 30, 479, 522, 90],
[39, 0, 45, 20, 567, 624, 90],
[40, 35, 30, 10, 264, 321, 90],
[41, 35, 32, 10, 166, 235, 90],
[42, 33, 32, 20, 68, 149, 90],
[43, 33, 35, 10, 16, 80, 90],
[44, 32, 30, 10, 359, 412, 90],
[45, 30, 30, 10, 541, 600, 90],
[46, 30, 32, 30, 448, 509, 90],
[47, 30, 35, 10, 1054, 1127, 90],
[48, 28, 30, 10, 632, 693, 90],
[49, 28, 35, 10, 1001, 1066, 90],
[50, 26, 32, 10, 815, 880, 90],
[51, 25, 30, 10, 725, 786, 90],
[52, 25, 35, 10, 912, 969, 90],
[53, 44, 5, 20, 286, 347, 90],
[54, 42, 10, 40, 186, 257, 90],
[55, 42, 15, 10, 95, 158, 90],
[56, 40, 5, 30, 385, 436, 90],
[57, 40, 15, 40, 35, 87, 90],
[58, 38, 5, 30, 471, 534, 90],
[59, 38, 15, 10, 651, 740, 90],
[60, 35, 5, 20, 562, 629, 90],
[61, 50, 30, 10, 531, 610, 90],
[62, 50, 35, 20, 262, 317, 90],
[63, 50, 40, 50, 171, 218, 90],
[64, 48, 30, 10, 632, 693, 90],
[65, 48, 40, 10, 76, 129, 90],
[66, 47, 35, 10, 826, 875, 90],
[67, 47, 40, 10, 12, 77, 90],
[68, 45, 30, 10, 734, 777, 90],
[69, 45, 35, 10, 916, 969, 90],
[70, 95, 30, 30, 387, 456, 90],
[71, 95, 35, 20, 293, 360, 90],
[72, 53, 30, 10, 450, 505, 90],
[73, 92, 30, 10, 478, 551, 90],
[74, 53, 35, 50, 353, 412, 90],
[75, 45, 65, 20, 997, 1068, 90],
[76, 90, 35, 10, 203, 260, 90],
[77, 88, 30, 10, 574, 643, 90],
[78, 88, 35, 20, 109, 170, 90],
[79, 87, 30, 10, 668, 731, 90],
[80, 85, 25, 10, 769, 820, 90],
[81, 85, 35, 30, 47, 124, 90],
[82, 75, 55, 20, 369, 420, 90],
[83, 72, 55, 10, 265, 338, 90],
[84, 70, 58, 20, 458, 523, 90],
[85, 68, 60, 30, 555, 612, 90],
[86, 66, 55, 10, 173, 238, 90],
[87, 65, 55, 20, 85, 144, 90],
[88, 65, 60, 30, 645, 708, 90],
[89, 63, 58, 10, 737, 802, 90],
[90, 60, 55, 10, 20, 84, 90],
[91, 60, 60, 10, 836, 889, 90],
[92, 67, 85, 20, 368, 441, 90],
[93, 65, 85, 40, 475, 518, 90],
[94, 65, 82, 10, 285, 336, 90],
[95, 62, 80, 30, 196, 239, 90],
[96, 60, 80, 10, 258, 318, 90],
[97, 50, 75, 20, 700, 760, 90],
[98, 47, 70, 10, 575, 626, 90],
[99, 40, 70, 20, 75, 106, 90],
[100, 40, 80, 10, 750, 801, 90],
]

# Define the header for the DataFrame
header = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']

# Create the DataFrame
df = pd.DataFrame(c101, columns=header)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)