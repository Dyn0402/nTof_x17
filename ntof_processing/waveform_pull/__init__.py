"""Raw n_TOF waveforms in the neighbourhood of every DREAM trigger.

See README.md. The short version: the slim says which (bunch, channel, time)
to look at, this reads the raw stream1 and keeps every zero-suppressed block
that overlaps, and the product sits beside the slim it was built from.
"""
