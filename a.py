# İlk olarak, 1-10 arasındaki sayıları içeren bir liste oluşturalım.
liste = []

print(liste)

# Sonra, bu listeye 2 değer daha ekleyelim.
liste.append(1)
liste.append(2)
liste.append(2)
liste.append(2)
liste.append(3)
liste.append(3)

# Son olarak, listenin her bir elemanını ve o elemanın sayısını ekrana yazdıralım.
for eleman in set(liste):
    print(f"{eleman}: {liste.count(eleman)}")

en_fazla_eleman = max(set(liste), key=liste.count)
en_fazla_sayi = liste.count(en_fazla_eleman)

print(f"en_fazla_eleman {en_fazla_eleman}")
print(f"en_fazla_sayi {en_fazla_sayi}")