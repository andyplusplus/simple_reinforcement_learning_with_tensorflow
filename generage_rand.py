import random


def get_rnd_str():
  ii = random.randint(10000, 99999)
  ss = str(ii)
  return ss

for x in range(500):
  s = ""
  for y in range(9):
    s += get_rnd_str() + " "
  s += get_rnd_str()
  print(s)