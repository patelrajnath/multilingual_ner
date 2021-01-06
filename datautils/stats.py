from statistics import mean

with open('../data/conll2003/conll2003_train_text.txt', encoding='utf8') as ftrain:
# with open('../data/conll2003/conll2003_testa_text.txt', encoding='utf8') as ftrain:
# with open('../data/alliance/alliance_train_text.txt', encoding='utf8') as ftrain:
# with open('../data/wallet/wallet_train_text.txt', encoding='utf8') as ftrain:
# with open('../data/nlu/nlu_train_text.txt', encoding='utf8') as ftrain:
# with open('../data/snips/snips_train_text.txt', encoding='utf8') as ftrain:
    main_dict = {}
    for item in ftrain:
        print()
        item1 = list(item.split(" "))
        print(item1)
        item2 = [' '.join(item1)]
        Length = [len(item1)]
        mydict = dict(zip(item2, Length))
        main_dict.update(mydict)

    print('Maximum Value: ', max(main_dict.values()))
    print('Minimum Value: ', min(main_dict.values()))
    print('average Value: ', mean(main_dict.values()))
