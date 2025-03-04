(c) 2024, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Discussion thread: https://bitcointalk.org/index.php?topic=5517607

<b>RCKangaroo-Fork Additions</b>
The only changes I have made is to how the program interacts with the command line. Now, all you have to enter is the DP, Range, and Public Key. The program will auto detect the offset and bit range, based off of the -range entered. 
Example:
```
RCKangaroo.exe -dp 14 -range 200000000000000000:400000000000000000 -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483
```

Results:
Example:
```
This software is free and open-source: https://github.com/RetiredC
It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP
Windows version
Start Range: 200000000000000000
End   Range: 400000000000000000
Bits: 70
CUDA devices: 1, CUDA driver/runtime: 12.6/12.6
GPU 0: NVIDIA GeForce RTX 4090, 23.99 GB, 128 CUs, cap 8.9, PCI 1, L2 size: 73728 KB
Total GPUs for work: 1

MAIN MODE

Solving public key
X: 90E6900A58D33393BC1097B5AED31F2E4E7CBD3E5466AF958665BC0121248483
Y: D7319F127105F492FD15E009B103B4A83295722F28F07C95F9A5443EF8E77CE0
Offset: 200000000000000000

Solving point: Range 70 bits, DP 14, start...
SOTA method, estimated ops: 2^35.202, RAM for DPs: 0.277 GB. DP and GPU overheads not included!
Estimated DPs per kangaroo: 3.067. DP overhead is big, use less DP value if possible!
GPU 0: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPUs started...
MAIN: Speed: 8090 MKeys/s, Err: 0, DPs: 2350K/2411K, Time: 0d:00h:00m:05s/0d:00h:00m:04s

Stopping work ...
Total Time: 7 seconds
Point solved, K: 1.373 (with DP and GPU overheads)


PRIVATE KEY: 349B84B6431A6C4EF1
```

I have tested it up to 100 bits, results:
```
This software is free and open-source: https://github.com/RetiredC
It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP
Linux version
Start Range: 000000000000000000000008000000000000000000000000
End   Range: 00000000000000000000000fffffffffffffffffffffffff
Bits: 99
CUDA devices: 8, CUDA driver/runtime: 12.4/12.0
GPU 0: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 1, L2 size: 73728 KB
GPU 1: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 65, L2 size: 73728 KB
GPU 2: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 98, L2 size: 73728 KB
GPU 3: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 129, L2 size: 73728 KB
GPU 4: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 161, L2 size: 73728 KB
GPU 5: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 193, L2 size: 73728 KB
GPU 6: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 194, L2 size: 73728 KB
GPU 7: NVIDIA GeForce RTX 4090, 23.64 GB, 128 CUs, cap 8.9, PCI 225, L2 size: 73728 KB
Total GPUs for work: 8

MAIN MODE

Solving public key
X: 4ECC524F1F53F525A7224364A4290BA97D72298D885FCF93B6E139E802B421B9
Y: 77621A8FCABAD9A502611EBB502359CE874065C1D0F5AF246028B38545B8990A
Offset: 0000000000000000000000000000000000000008000000000000000000000000

Solving point: Range 99 bits, DP 24, start...
SOTA method, estimated ops: 2^49.702, RAM for DPs: 2.220 GB. DP and GPU overheads not included!
Estimated DPs per kangaroo: 8.674.
GPU 0: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 1: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 2: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 3: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 4: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 5: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 6: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPU 7: allocated 2394 MB, 786432 kangaroos. OldGpuMode: No
GPUs started...
MAIN: Speed: 59748 MKeys/s, Err: 0, DPs: 37036K/54571K, Time: 0d:02h:54m:38s/0d:04h:15m:23s

Stopping work ...
Total Time: 2 hours, 54 minutes, 41 seconds
Point solved, K: 0.781 (with DP and GPU overheads)


PRIVATE KEY: 000000000000000000000000000000000000000F4A21B9F5CE114686A1336E07
```

Windows Release in Release section was built with Cuda 12.6 props and compute_89,sm_89;compute_86,sm_86;compute_75,sm_75;compute_61,sm_61. Should work on most current GPUs. Or you can download and compile on your own.

<b>Features:</b>

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fast, about 8GKeys/s on RTX 4090, 4GKeys/s on RTX 3090.
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.

<b>Limitations:</b>

- No advanced features like networking, saving/loading DPs, etc.

<b>Command line parameters:</b>

<b>-gpu</b>		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

<b>-pubkey</b>		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

<b>-start</b>		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

<b>-range</b>		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

<b>-dp</b>		DP bits. Must be in range 14...60. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

<b>-max</b>		option to limit max number of operations. For example, value 5.5 limits number of operations to 5.5 * 1.15 * sqrt(range), software stops when the limit is reached. 

<b>-tames</b>		filename with tames. If file not found, software generates tames (option "-max" is required) and saves them to the file. If the file is found, software loads tames to speedup solving. 

When public key is solved, software displays it and also writes it to "RESULTS.TXT" file. 

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Sample command to generate tames:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10

Then you can restart software with same parameters to see less K in benchmark mode or add "-tames tames76.dat" to solve some public key in 76-bit range faster.

<b>Some notes:</b>

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 


<b>Changelog:</b>

v3.0:

- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:

- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:

- added ability to start software on 30xx cards.

v1.0:

- initial release.
