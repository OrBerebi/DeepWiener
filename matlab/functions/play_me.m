function play_me(p_t,Fs)
    player = audioplayer(p_t, Fs);
    play(player)
    pause((size(p_t,2)/Fs)*1.2)
end