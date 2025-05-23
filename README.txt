# P1 NOTES

Label classes
1 = clouds
2 = Water
3 = Vegetation
4 = bare earth
5 = glacier / ice
7 = cloud shadows

# PHASE 1 - REVIEW
# After the first run of the model with 87 training points, water was noticeably being classified in areas shadowed by the mountains.
# The performance of vegetation and bare earth was excellent
# Some of the edges around clouds turned into glacier
# Cloud shadows were quite disruptive to the model


# PHASE 2 - UPDATES
# The updates made were: new class for cloud shadows, more samples taken at edge of clouds, and more samples of vegetation in shadows behind mountains

# PHASE 2 - REVIEW
# The cloud edges got better
# A lot of the water that was actually mountain-shadowed vegetation saw great improvements, though some water was lost 
	# (there were very few training points for this to begin with)
# Bare earth looks really great
# Glaciers and clouds are well delineated even though they look similar in regular RGB view

# PHASE 3 - Water Update
# Seeing some major losses to water, I labeled a large number of additional water points, including and especially those underneath light clouds and aerosols

# PHASE 3 - REVIEW
# Now the water seems to be indicating a lot of vegetated areas to be slivers of water. This is likely not a training data problem, but a model problem.


# PHASE 4 - Random Forest
# I upgraded the model to a RF with 100 estimators. 
# This showed great improvements and finally captured the full extent of the lake
# However, some of the mountain shadows are again classifying as lakes

# PHASE 5 - XGBoost
# I upgraded the model once more to XGBoost and saw even more fine-tuned improvements with bare earth crests
# Although the mountain shadows returned, they are far better than the RF model

# PHASE 6 - More training data
# I added even more training data for the shadowed mountain areas

# PHASE 6 - REVIEW
# The addition of more training points really reduced the number of watered mountain shadows. Looks great
# The only remaining item are cloud shadows looking like water now too
