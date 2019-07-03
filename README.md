# intrinsic-images


# Explanation of data directory

* data
    * mmap; material maps
        * **mmap_npy**; 512x512 material maps in numpy array form
        * **mmap_real_cr2**; Canon raw, uncropped images of material maps
        * **mmap_real_npy**; uncropped raw material images converted into numpy arrays from 16-bit (linear) tiff
        * **mmap_real_tiff**; 16-bit linear tiffs of the Canon raw images
        * **mmap_synthetic**; 512x512 ppms generated using Bruce's C code
    * imap; illumination map
        * **imap_npy**; 512x512 illumination maps
        * **imap_npy_ambient**; 512x512 ambient illumination maps
        * **imap_npy_direct**; 512x512 direct illumination maps
        * **imap_real_cr2**; Canon raw, uncropped images of illumination maps
        * **imap_real_npy**; uncropped raw illumination images converted into numpy arrays from 16-bit (linear) tiff
        * **imap_real_tiff**; 16-bit linear tiffs of the Canon raw images
