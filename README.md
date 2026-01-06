# Synergistic VIS-DSM Learning for Deepfake Detection and Localization in Satellite Imagery

The code and datasets for Synergistic VIS-DSM Learning for Deepfake Detection and Localization in Satellite Imagery.

# Data preprocessing

Our datasets could be obtained from link: https://pan.baidu.com/s/1l5HUmTVScgO8OzwSlvLIxw?pwd=mdmd. Password：mdmd (./Fake-Vaihingen-MM & Fake-Potsdam-MM.zip)

Put them into ./data.

```
FAKE-VAIHINGEN-MM & FAKE-POTSDAM-MM

+---Fake-Potsdam-MM
|   +---Fake-Potsdam-DSM
|   |   +---test
|   |   |   +---gt_dsm
|   |   |   +---gt_generated_place
|   |   |   +---inpainted_mask
|   |   |   +---lama_gt
|   |   |   +---lama_mask
|   |   |   +---repaint_gt
|   |   |   \---repaint_mask
|   |   \---train
|   |       +---gt_dsm
|   |       +---gt_generated_place
|   |       +---inpainted_mask
|   |       +---lama_gt
|   |       +---lama_mask
|   |       +---repaint_gt
|   |       \---repaint_mask

+---Fake-Potsdam-MM
|   \---Fake-Potsdam-VIS
|       +---gt_mask
|       +---test
|       |   +---gt
|       |   +---gt_mask
|       |   +---inpainted
|       |   +---inpainted_mask
|       |   +---lama
|       |   +---lama_gt
|       |   +---lama_mask
|       |   +---repaint
|       |   +---repaint_gt
|       |   \---repaint_mask
|       \---train
|           +---gt
|           +---gt_mask
|           +---inpainted
|           +---inpainted_mask
|           +---lama
|           +---lama_gt
|           +---lama_mask
|           +---repaint
|           +---repaint_gt
|           \---repaint_mask

\---Fake-Vaihingen-MM
    +---Fake-Vaihingen-DSM
    |   +---gt
    |   +---gt_mask
    |   +---test
    |   |   +---gt_dsm
    |   |   +---gt_generated_place
    |   |   +---inpainted_gray
    |   |   +---inpainted_mask
    |   |   +---inpainted_mask_gray
    |   |   +---inpainted_rgb
    |   |   +---lama_gray
    |   |   +---lama_gt
    |   |   +---lama_mask
    |   |   +---lama_rgb
    |   |   +---repaint_gray
    |   |   +---repaint_gt
    |   |   +---repaint_mask
    |   |   \---repaint_rgb
    |   \---train
    |       +---gt_draft
    |       +---gt_dsm
    |       +---gt_geanerated_place
    |       +---inpainted_gray
    |       +---inpainted_mask
    |       +---inpainted_mask_gray
    |       +---inpainted_rgb
    |       +---lama_gray
    |       +---lama_gt
    |       +---lama_mask
    |       +---lama_rgb
    |       +---repaint_gray
    |       +---repaint_gt
    |       +---repaint_mask
    |       \---repaint_rgb

\---Fake-Vaihingen-MM
    \---Fake-Vaihingen-VIS
        +---test
        |   +---gt
        |   +---gt_generated_place
        |   +---inpainted
        |   +---inpainted_mask
        |   +---lama
        |   \---repaint
        \---train
            +---gt
            +---inpainted
            +---inpainted_mask
            +---lama
            \---repaint
```

# Training and testing
<!-- Training and testing  -->
- Train

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u src/main.py --dsm_option True --data_train Vaihingen --data_train_dir fakeV --model FLDCF_multiModal --save FLDCF_multiModal__Fake_Vaihingen__NIRRG_withGTDSM > FLDCF_multiModal__Fake_Vaihingen__NIRRG_withGTDSM 2>&1 &
```

<!-- python src/main.py --data_train Vaihingen --data_train_dir fakeV --model fldcf -->


- Test
```bash
python src/test.py --dsm_option True --data_train Vaihingen --data_train_dir fakeV  --model FLDCF_multiModal --pre_train './model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt' --save FLDCF_multiModal__Fake_Vaihingen__NIRRG_withGTDSM__Test
```

<!-- ./model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt -->

# Trained checkpoints

Our pretrain models could be obtained from link: https://pan.baidu.com/s/1l5HUmTVScgO8OzwSlvLIxw?pwd=mdmd. Password：mdmd (./MFLDCF_models.zip)

# BibTeX
Will be available soon.

