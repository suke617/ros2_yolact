# ros2_yolact

yolactをros2で用いるためのツール


yolacのメッセージ型

yolactのマスクは一次元配列に戻して送っているようなので
np.unpackbitsを用いて元の配列の形に戻してやる必用がある
```
mask_msg=msg.detections[i].mask
box=msg.detections[i].box
unpacked_mask = np.unpackbits(mask_msg.mask, count=mask_msg.height*mask_msg.width)
unpacked_mask = unpacked_mask.reshape((mask_msg.height, mask_msg.width))
```

