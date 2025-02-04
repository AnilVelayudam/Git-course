CP = {"Sachin","Rohit","Virat"}
TP ={"Virat","Rossum","Travis"}

#Find all the players who are playing all the games.
player=CP.union(TP)
print(player)

#Find all the players who are playing Both Circket and Tennis--
player1=CP.intersection(TP)
print(player1)

#) Find all the players who are playing Only Cricket but not tennis--
player2=CP.difference(TP)
print(player2)

#Find all the players who are playing Only tennis but not cricket
player3=TP.difference(CP)
print(player3)

#Find all the players who are playing Exclusively Cricket and Tennis----
player4=TP.symmetric_difference(CP)
print(player4)




