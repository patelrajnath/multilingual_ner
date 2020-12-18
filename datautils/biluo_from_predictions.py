def get_biluo(bio_tags):
    biluo_tags = []
    i = 0
    last_tag = None
    while i < len(bio_tags):
        item = bio_tags[i]

        if item == 'O':
            biluo_tags.append(item)
            last_tag = item
            i += 1
            continue

        suffix = item[2:]

        # In case the input text has the B-O notations
        if suffix == 'O':
            biluo_tags.append('O')
            last_tag = item
            i += 1
            continue

        suffix_next_tag = None
        suffix_last_tag = None

        if last_tag:
            suffix_last_tag = last_tag[2:]

        if i + 1 < len(bio_tags):
            suffix_next_tag = bio_tags[i + 1][2:]

        if suffix_last_tag == suffix:
            if suffix_next_tag:
                if suffix == suffix_next_tag:
                    biluo_tags.append('I-' + suffix)
                else:
                    biluo_tags.append('L-' + suffix)
            else:
                biluo_tags.append('L-' + suffix)
        else:
            if suffix_next_tag:
                if suffix == suffix_next_tag:
                    biluo_tags.append('B-' + suffix)
                else:
                    biluo_tags.append('U-' + suffix)
            else:
                biluo_tags.append('U-' + suffix)
        i += 1
        last_tag = item
    return biluo_tags


if __name__ == '__main__':
    # bio = "B_O B_O B_O B_O B_hms B_O B_O B_user I_user B_O B_user I_user".split()
    bio = ['O', 'O', 'O', 'O', 'U-hms', 'O', 'O', 'B-user', 'L-user', 'O', 'I-user']
    print(get_biluo(bio))
