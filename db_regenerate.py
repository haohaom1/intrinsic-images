import sys, os

'''

    Allen Ma Summer 2019
    Delete all imaps whose filenames start with the prefixes listed in to_change_prefixes

'''

def main(argv):
    if len(argv) < 2:
        print("python3 db_regenerate.py <path_imap>- for path_imap, pass in the base path like imap_npy/")

    path_imap = argv[1]

    ambient_path_imap = path_imap + "_ambient"
    direct_path_imap = path_imap + "_direct"

    if not os.path.isdir(ambient_path_imap):
        print(ambient_path_imap, " is not a valid path")

    if not os.path.isdir(direct_path_imap):
        print(direct_path_imap, " is not a valid path")

    to_change_prefixes = ["fractal", "perlin", "random", "stripe"]

    dirs = [ambient_path_imap, direct_path_imap, path_imap]
    subdirs = ["train", "test"]

    for d in dirs:
        for subdir in subdirs:
            count = 0
            full_dir = os.path.join(d, subdir)
            assert(os.path.isdir(full_dir))
            for imap_name in os.listdir(full_dir):
                if any(imap_name.lower().startswith(x) for x in to_change_prefixes):
                    fullname = os.path.join(full_dir, imap_name)
                    os.remove(fullname)
                    print(f"removed {fullname}")
                    count += 1
            print(f"removed {count} from {d}/{subdir}")
        
            





if __name__ == "__main__":
    main(sys.argv)