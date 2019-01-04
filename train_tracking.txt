Inception fine tune indices = [311, 299, 294, 292]
  -lrsmall = [optimizer(lr=0.0006), optimizer(lr=0.0001)]
  -lrsmed = [optimizer(lr=0.006), optimizer(lr=0.001)] (SIMILAR TO SMALL, A LITTLE WORSE)
  -lrlarge = [optimizer(lr=0.06), optimizer(lr=0.01)] (NOT GOOD)

  -aug1 = augmentation strength = 0.1
  -aug2 = augmentation strength = 0.2
  -aug3 = augmentation strength = 0.3

  =FT2 = fine tune base with 1 additional layer: [311, 299]
  =FT3 = fine tune base with 2 additional layers: [311, 299, 294]
  =FT4 = fine tune base with 3 additional layers: [311, 299, 294, 292]

  inception_pad_FT4_Adam_aug1_lrsmall = holdout loss: 0.6359473326191398 accuracy: 0.670807465263035

  inception_pad_FT3_Adam_aug1_lrsmall = holdout loss: 0.6419479746996246 accuracy: 0.6708074613757755

  inception_orig_FT3_Adam_aug1_lrsmall = holdout loss: 0.6403521443746105 accuracy: 0.6770186539022078

  Final evaluated accuracy on just neural net: holdout loss: 0.6413683071732521 accuracy: 0.6750000044703484
