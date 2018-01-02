img_dim = 100 * 100
scales = []


for k in range(1,21):
	scale = 12 / ( img_dim / 20  + img_dim * (k - 1) / 20)
	scales.append(scale)
	if scale * img_dim < 12:
		break

print(scales)
