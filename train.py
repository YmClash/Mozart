import pretty_midi
import numpy as np
import midiutil
import mido




midi_to_roll = pretty_midi.PrettyMIDI(r"MIDI/amir_midi.mid")

midi = mido.MidiFile(r"MIDI/Caribbean-Blue.mid",clip=True)

allo = midi.tracks
for i in allo[3][:20]:
    print(i)


numb = midi_to_roll.lyrics
print(numb)

print(allo)


# end_time = int(midi_to_roll.get_end_time())
#
# combine_piano_roll = np.zeros((128, end_time))
#
# for intru in midi_to_roll.instruments:
#     piano_roll = intru.get_piano_roll(fs=100)
#     combine_piano_roll = np.maximum(combine_piano_roll,piano_roll)
#
#
#
#
# print(combine_piano_roll.shape)
#
#
#

