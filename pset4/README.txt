# eseitz - Emily Seitz
# 3/5/12
# 6.815 A4

* 9 hrs.
* I implemented everything, and it runs.  I do see a problem, however, with my greenBasedDemosaick-ing.  The top and left edges have a hint of blue, while the bottom and rright edges have a hint of red.  I think it's a logical error in my interpolation of the red/blue channels, but I haven't been able to find the bug :(
* none
* Worked on my own while consulting the Piazza forum.
* I was confused at first about how to implement the green-based red and blue, mostly about why we were subtracting the green channel.  But once I tried it and saw the result... it made sense :)
* RAW --> RGB is super cool!  I've been showing the magic to my neighbors and trying to explain the concept to them.
* In my results, the SNR for 400 is pretty dark and the SNR for 3200 is a little lighter.  Therefore, the 3200 has better SNR.  (I expected 400, though)
* I interpolated along the direction with the lowest difference format the edge-based demosaicking.

