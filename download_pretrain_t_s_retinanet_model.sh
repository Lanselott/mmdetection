wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1urVb40bjsgrvcax90oeohDCjsUXOxDTZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1urVb40bjsgrvcax90oeohDCjsUXOxDTZ" -O 'retinanet_t_s_finetune_student_from_scratch_epoch_12.pth' && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/1urVb40bjsgrvcax90oeohDCjsUXOxDTZ/view?usp=sharing