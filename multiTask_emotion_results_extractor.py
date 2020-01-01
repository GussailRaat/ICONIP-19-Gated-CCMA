#####################################
Preference F1 score
#####################################
echo "---------------Emotion-Multi-task Preference F1--------------" >> Emotion-Multi-task.txt
echo "====Uni- Text - Audio - Video====" >> Emotion-Multi-task.txt
cat emotion_unimodal_text_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 6,6 | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_unimodal_audio_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_unimodal_video_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

echo "====Bi-MMMU-BA - TV - TA - AV====" >> Emotion-Multi-task.txt
cat emotion_bimodal_text_video_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_bimodal_text_audio_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_bimodal_audio_video_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

echo "====Tri-MMMU-BA - MMUU-SA - Bi-MU-SA====" >> Emotion-Multi-task.txt
cat emotion_trimodal_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt


#####################################
Preference W-Acc
#####################################
echo "---------------Emotion-Multi-task Preference W-Acc--------------" >> Emotion-Multi-task.txt
echo "====Uni- Text - Audio - Video====" >> Emotion-Multi-task.txt
cat emotion_unimodal_text_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_unimodal_audio_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_unimodal_video_True:80_True:10.txt | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

echo "====Bi-MMMU-BA - TV - TA - AV====" >> Emotion-Multi-task.txt
cat emotion_bimodal_text_video_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_bimodal_text_audio_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt
cat emotion_bimodal_audio_video_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

echo "====Tri-MMMU-BA - MMUU-SA - Bi-MU-SA====" >> Emotion-Multi-task.txt
cat emotion_trimodal_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt


