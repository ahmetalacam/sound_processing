"""
import os

# Dosyaların bulunduğu dizini belirtin
directory = 'path/to/your/directory'

# Eski ve yeni isimleri belirleyin
old_string = 'fear'
new_string = 'FEA'

# Dizindeki tüm dosyaları gezin
for filename in os.listdir(directory):
    # Eski stringi içeren dosya adlarını bulun
    if old_string in filename:
        # Yeni dosya adını oluşturun
        new_filename = filename.replace(old_string, new_string)
        # Eski ve yeni dosya yollarını oluşturun
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        # Dosyayı yeniden adlandırın
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')
"""



import os

# Dosyaların bulunduğu dizini belirtin
directory = 'path/to/your/directory'

# Eklenecek string
suffix = '_X'

# Dizindeki tüm dosyaları gezin
for filename in os.listdir(directory):
    # Sadece .wav uzantılı dosyaları işlem yapalım
    if filename.endswith('.wav'):
        # Yeni dosya adını oluşturun
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{ext}"
        # Eski ve yeni dosya yollarını oluşturun
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        # Dosyayı yeniden adlandırın
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')

