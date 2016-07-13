# ParallelStaticObjectDetection
Detect New Static Object in a already defined background using Parallel programming.
Objective: To detect new static objects in an already defined static background.
Main Design:
Send initial nackground frame to all the nodes.
Then send incoming frames to the worker nodes in the order of incoming frames.
Each worker nodes compute the frame difference for Red, Greenm Blue and RGB combined image
Result is then send to collector nodes.
