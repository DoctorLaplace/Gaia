WEBVTT

1
00:00:01.740 --> 00:00:09.140
Matt Clark: Okay… Hey everybody, let me share… that is reshared.

2
00:00:12.160 --> 00:00:13.290
Matt Clark: Okay.

3
00:00:13.550 --> 00:00:21.110
Matt Clark: So, I'm giving a, presentation for a capstone project, on Professor Clark in,

4
00:00:21.490 --> 00:00:24.449
Matt Clark: The Geography, Environment, and Planning program.

5
00:00:24.730 --> 00:00:36.359
Matt Clark: And I teach our classes in geographic information systems, advanced GIS, Environmental Remote Sensing, which is what this project's gonna be focused on.

6
00:00:36.540 --> 00:00:40.049
Matt Clark: And kind of ecology and field classes.

7
00:00:41.990 --> 00:00:56.269
Matt Clark: And, so this is gonna fit into a class that I am teaching this semester, which is called Environmental Remote Sensing. And remote sensing is the use of a sensor to measure some part of the Earth's surface.

8
00:00:56.450 --> 00:01:05.229
Matt Clark: We're not in direct contact, so it could be from a drone, it could be from a helicopter, airplane, satellite, on the space station.

9
00:01:06.110 --> 00:01:12.930
Matt Clark: And then you're turning that, information that you're recording with a sensor into a useful map.

10
00:01:13.110 --> 00:01:22.360
Matt Clark: They can then be used in a geographic information system or other kind of application. Anyways, tons of applications fit into remote sensing.

11
00:01:22.470 --> 00:01:29.430
Matt Clark: And, we use many different kinds of sensors. It's not just color photos.

12
00:01:30.820 --> 00:01:47.280
Matt Clark: One of the workhorse sensors that we use in remote sensing is called Landsat. There's… we're now on Landsat 9. There's been, multiple versions of this satellite since the 70s, and so now provides a nice 50-year record of the Earth and different kinds of land change.

13
00:01:48.970 --> 00:02:03.030
Matt Clark: It is what we call a multispectral satellite, which measures, certain parts of the electromagnetic spectrum, so, like, red, green, blue, near-infrared, and parts of the shortwave infrared, part of the light spectrum.

14
00:02:03.350 --> 00:02:18.220
Matt Clark: And, I'll talk a little more about that in a minute, but, because it has so, been up there for so long, and, it basically comes around every 16 days on any spot on the Earth, you've got this nice temporal record.

15
00:02:18.400 --> 00:02:20.719
Matt Clark: And that we can use…

16
00:02:20.870 --> 00:02:28.659
Matt Clark: Not just to look at an image, you know, at one point in time, but actually look at how things have changed over time, as you see in this image here.

17
00:02:31.440 --> 00:02:47.590
Matt Clark: So there's… right currently, there's two Landsat satellites, in operation. These are managed by NASA, Landsat 8, Landsat 9. They… this is the electromagnetic spectrum of, visible lights over here. This is what our eyes see.

18
00:02:48.020 --> 00:02:57.400
Matt Clark: In this part of the spectrum, and this is what we call near-infrared, and we call these regions shortwave infrared, and out here we have thermal energy.

19
00:02:58.040 --> 00:03:06.960
Matt Clark: The Landsat satellites are multi-spectral in that they're measuring little slices, little parts of the spectrum in just a few regions.

20
00:03:07.400 --> 00:03:12.389
Matt Clark: The Europeans have, two satellites called Sentinel-2A and B.

21
00:03:12.600 --> 00:03:21.639
Matt Clark: And they're also multispectral, but they have additional measurements of this area here that we call the red edge, between red light and near-infrared.

22
00:03:21.780 --> 00:03:27.840
Matt Clark: And… but they're fairly similar, as you can see, like, where they're taking measurements in the spectrum.

23
00:03:28.220 --> 00:03:42.939
Matt Clark: So those are both multispectral satellites, and there's been an effort to kind of harmonize the two so that we get, basically, more looks on the Earth and more temporal data by combining them together into one, kind of product.

24
00:03:43.090 --> 00:03:46.730
Matt Clark: And so this is called the Harmonized Landsat Sentinel-2 dataset.

25
00:03:49.470 --> 00:03:50.800
Matt Clark: And,

26
00:03:51.070 --> 00:03:57.480
Matt Clark: This is a new way of doing remote sensing that I want to explore in this capstone project.

27
00:03:57.780 --> 00:04:03.999
Matt Clark: And it's using transformers, in this vision, transformer architecture.

28
00:04:04.520 --> 00:04:12.540
Matt Clark: Where… and I don't… this is where I, you know, the computer science is beyond my knowledge as a remote sensor, but I'm interested.

29
00:04:12.710 --> 00:04:17.660
Matt Clark: in how it's… how it can be used. So NASA teamed up with IBM,

30
00:04:18.079 --> 00:04:33.139
Matt Clark: A couple years ago, and they created this foundational model called Prithvi. They first had 1.0. It was basically, trained, developed on this Landsat Sentinel harmonized dataset, the HLS dataset.

31
00:04:33.430 --> 00:04:43.969
Matt Clark: And it looks spectrally, but also spatially, and through time. So the beauty of it is it's spatial, temporal, and spectral, so different parts of the spectrum.

32
00:04:44.420 --> 00:04:56.299
Matt Clark: And the way I understand it is you basically have some areas that are masked out of the image, and it tries to learn the features, basically encode features.

33
00:04:56.340 --> 00:05:08.189
Matt Clark: That could then be used to, decode and fill in missing parts of the image. So they purposely ma-, as you see here, purposely mask

34
00:05:08.420 --> 00:05:21.689
Matt Clark: parts of the image stack. And again, this is spectral-temporal, and doing this spatially as well. But it's learning, features without any kind of supervision.

35
00:05:21.960 --> 00:05:28.050
Matt Clark: That could then be used to basically fill in the missing images that were masked out.

36
00:05:28.780 --> 00:05:31.360
Matt Clark: And you get reconstructed images.

37
00:05:31.980 --> 00:05:47.700
Matt Clark: In that process, because there's no training data involved in this, it's just learning spectral-temporal, spatial features. It's interesting, because it's basically the machine is creating features that can be used for other applications.

38
00:05:48.030 --> 00:06:00.210
Matt Clark: And so that's, where I want to take this project, as you'll see. It's going to be using this transformer architecture. We're not going to use Prithvi, but it's an interesting approach to remote sensing that's starting to take

39
00:06:00.330 --> 00:06:02.149
Matt Clark: Get, some momentum.

40
00:06:02.620 --> 00:06:15.809
Matt Clark: They later, included also latitude and longitude to bring in kind of, like, where you are around the Earth. It's not just the spatial arrangement of pixels that matter, but also, you know, whether… what kind of ecosystem you're in.

41
00:06:16.170 --> 00:06:26.749
Matt Clark: You know, are you in a tropical rainforest, or are you in a California Mediterranean climate area? So that matters as well. So that was a… that was the 2.0, the kind of the big…

42
00:06:26.880 --> 00:06:32.279
Matt Clark: Innovation is also bringing in latitude, longitude, year, and day of year.

43
00:06:32.860 --> 00:06:44.790
Matt Clark: And so that gets kind of brought into the encoding framework and can be used to help… you can, like, leverage that information when you're doing the decoding and reconstructing the image.

44
00:06:45.270 --> 00:06:50.829
Matt Clark: And I think that would be interesting to bring into our analysis, too. Okay, so,

45
00:06:51.540 --> 00:07:09.849
Matt Clark: I just got this off of, you know, these GitHubs that NASA had, with some Python, Jupyter notebooks. Just to give you an example of how these… so, in the process of these transformers, you get embeddings of the spatial-temporal spectral features.

46
00:07:10.090 --> 00:07:22.439
Matt Clark: And those can be used for one… one application is to use those to basically do classification. So here's crop classification, where this is what the ground truth looks like.

47
00:07:22.680 --> 00:07:29.450
Matt Clark: And, maybe you could, like, classify crops at different time periods, for example.

48
00:07:29.850 --> 00:07:34.480
Matt Clark: I just want to… that's a classic remote sensing application, is to try and do…

49
00:07:34.590 --> 00:07:37.530
Matt Clark: Classification, building a land cover map.

50
00:07:38.670 --> 00:07:50.619
Matt Clark: This one was a segmentation-type, example, where they're using this to show landslides in kind of image segmentation, so just another example.

51
00:07:50.740 --> 00:07:54.600
Matt Clark: And this one's kind of closer to where I want to take this project.

52
00:07:55.420 --> 00:08:05.809
Matt Clark: Which is regression. So regression is kind of like in a simple linear model, where you've got one predictor variable and you're trying to predict something else. In this case, you might have, like.

53
00:08:06.000 --> 00:08:12.669
Matt Clark: Global, this is gross primary productivity.

54
00:08:12.960 --> 00:08:18.550
Matt Clark: It's the truth from the ground, where they kind of think they have, like, the truth, and then…

55
00:08:18.680 --> 00:08:26.089
Matt Clark: through this, these Prithvi features, they've predicted, GPP.

56
00:08:26.650 --> 00:08:36.050
Matt Clark: And then you can, you know, look at these as, like, a 2D scatter plot, and fit the line across that, and that's a pretty high R-squared .94.

57
00:08:36.260 --> 00:08:44.779
Matt Clark: This, this R squared is a regression metric that it goes from 0 to 1, 1 being closer to the straight line.

58
00:08:45.040 --> 00:08:51.070
Matt Clark: And, you know, further… when there's more scattered, the R-square goes down.

59
00:08:52.360 --> 00:09:02.549
Matt Clark: So remember this particular application of trying to do a regression, because that's what I want to do, is… in my case, it's going to be species richness instead of…

60
00:09:02.920 --> 00:09:05.510
Matt Clark: Global primary productivity.

61
00:09:07.320 --> 00:09:16.910
Matt Clark: Okay, so this moves us to, the application, which is coming from a project, a NASA project that I'm part of called Bioscape.

62
00:09:17.260 --> 00:09:30.149
Matt Clark: And this is a biodiversity survey of southern Africa and the Cape region. It's an area with, high species diversity. Some species only occur in that region.

63
00:09:30.360 --> 00:09:41.520
Matt Clark: There's multiple teams that have been selected by NASA to work on biodiversity applications, and my team is looking at, how can we, use NASA's

64
00:09:41.750 --> 00:09:49.430
Matt Clark: Advanced sensors to map animal species diversity.

65
00:09:52.930 --> 00:10:07.070
Matt Clark: And they came to South Africa and flew, 3 advanced sensors, in 2 different airplanes. So they actually brought these planes down to Africa, and they've got holes in the bottom, and they've got sensors pointed down,

66
00:10:07.210 --> 00:10:20.470
Matt Clark: The sensor that we're going to be really focused on is called, Airborne Visible Infrared Imaging Spectrometer, NextGen, which is AVRIS-NG. It's what we call a hyperspectral sensor, and it measures lots of parts of the spectrum of…

67
00:10:20.680 --> 00:10:23.789
Matt Clark: Reflected light off the ground.

68
00:10:24.840 --> 00:10:28.229
Matt Clark: There's other sensors too, but we're not gonna deal with those.

69
00:10:28.640 --> 00:10:32.689
Matt Clark: And these data were collected in October to December 2023.

70
00:10:35.360 --> 00:10:41.590
Matt Clark: So, just, to go off of, when I was talking about Landsat and Sentinel-2, those are multispectral sensors.

71
00:10:41.780 --> 00:10:52.890
Matt Clark: That are measuring just certain parts of the spectrum, just little measurements. And if you were to look at reflected light across wavelengths for different land cover types, it may look kind of…

72
00:10:53.300 --> 00:10:58.970
Matt Clark: Kind of blocky, because you only have a certain number of measurements across the spectrum.

73
00:10:59.090 --> 00:11:03.829
Matt Clark: Now, with hyperspectral data, which is what this Avarice NG sensor is.

74
00:11:03.940 --> 00:11:18.710
Matt Clark: You get hundreds of measurements across the wavelengths, and you get these beautiful spectra of reflected light, from visible light, what your eyes see, through near-infrared and shortwave infrared, energy.

75
00:11:19.020 --> 00:11:32.299
Matt Clark: So lots of measurements, and those data can be used to distinguish different species of plants. That's the primary application, is trying to do, like, species identification.

76
00:11:32.590 --> 00:11:38.210
Matt Clark: Of plants, but in our case, we're going to be looking at animal richness, diversity.

77
00:11:38.470 --> 00:11:48.360
Matt Clark: But that's linked to plant diversity. So the hypothesis is when there's more plant diversity, that begets more animal diversity.

78
00:11:48.720 --> 00:11:54.960
Matt Clark: And that plant diversity is then, you know, picked up by, spectral variability.

79
00:11:56.260 --> 00:12:00.939
Matt Clark: That's as encoded by this hyperspectral sensor, okay?

80
00:12:02.840 --> 00:12:14.770
Matt Clark: So, to get the animal diversity data, what we used is, these little sound recorders that just go, like, little circuit boards with batteries, an SD card, and a built-in mic.

81
00:12:15.030 --> 00:12:20.030
Matt Clark: That we put in these little waterproof boxes and just leave out for a few days. We about…

82
00:12:20.170 --> 00:12:23.470
Matt Clark: 4 to 2 weeks… 4 days to 2 weeks.

83
00:12:23.670 --> 00:12:31.649
Matt Clark: But we put them out, all around South Africa, the southern part of South Africa called the Greater Cape Floristic Region.

84
00:12:32.160 --> 00:12:48.369
Matt Clark: And these dots represent where we put out the sound recorders. We actually went out twice, once in the wet season and once in the dry season. That way, we could capture different kinds of birds, and if… we also looked… we're looking at frogs, and the frogs were more prevalent when there was water in the wet season.

85
00:12:49.620 --> 00:12:56.679
Matt Clark: But these gray boxes you see here is where NASA flew their hyperspectral sensor.

86
00:12:57.110 --> 00:13:02.589
Matt Clark: These other little black lines swirling around are where they flew a laser sensor called LiDAR.

87
00:13:04.230 --> 00:13:18.069
Matt Clark: What we did, this is a whole project, and it's done and complete, and we're going to use these data, is, for each of the sites where we left out the recorder, we built a convolutional neural network, species detector that works on the sound data.

88
00:13:18.190 --> 00:13:28.829
Matt Clark: to give you, different species of birds, frogs, and insects. And that's what we're going to use as our basis for, animal richness in this project.

89
00:13:29.420 --> 00:13:37.150
Matt Clark: So that's a whole other piece of work of, you know, dealing with, using AI to detect animals in sounds.

90
00:13:37.390 --> 00:13:49.429
Matt Clark: But, you know, it has a certain level of error, too. It's not perfect, but, the neat thing is, the sensors are out there for 24-7 for over a long period of time, and it gives you more opportunity.

91
00:13:49.520 --> 00:14:00.290
Matt Clark: to hear different animals and actually detect them, which is different than the typical approaches where you go out to a spot, maybe sit there for 20 minutes and try and hear and see

92
00:14:00.410 --> 00:14:03.040
Matt Clark: Different animals, and so that's more limited.

93
00:14:03.220 --> 00:14:07.180
Matt Clark: So there's, there's good things about this, kind of acoustic approach.

94
00:14:08.190 --> 00:14:18.069
Matt Clark: Even though it may not be sometimes as accurate as actually seeing that animal. If your animal doesn't make any noise, we don't even detect it.

95
00:14:19.140 --> 00:14:27.030
Matt Clark: Okay, so what we do… we've been doing in our research team Is taking the hyperspectral data.

96
00:14:27.640 --> 00:14:34.709
Matt Clark: That NASA collected, and basically created what we call engineered features. So these are features…

97
00:14:34.820 --> 00:14:40.429
Matt Clark: That are related to photosynthesis, structural.

98
00:14:40.600 --> 00:14:50.380
Matt Clark: Plant materials like lignin and cellulose, and also, productivity, like nitrogen, water content, so like the full year water content.

99
00:14:50.680 --> 00:14:58.429
Matt Clark: These are all, what we do is we create features that are related to absorptions that are seen in the reflected light.

100
00:14:58.570 --> 00:15:04.000
Matt Clark: And these come from the literature, some of them from my own, kind of homebrew things that I've created over the years.

101
00:15:04.190 --> 00:15:22.450
Matt Clark: But I've been doing this approach since I did my PhD dissertation in 2021. So, this is kind of an old school, if you want, standard approach, is to create, engineered features, you know, that are related to different biochemicals.

102
00:15:22.630 --> 00:15:24.600
Matt Clark: They come from the literature.

103
00:15:25.250 --> 00:15:43.309
Matt Clark: And they're fairly easy to compute, and, you know, nothing super sophisticated. But that's what we did, so that's kind of like our first cut. And then what we did is, relate these features to our, acoustic-based species richness.

104
00:15:43.680 --> 00:15:46.029
Matt Clark: Using a random forest model.

105
00:15:47.620 --> 00:15:51.410
Matt Clark: And we looked at different spatial scales, and this was, like.

106
00:15:51.790 --> 00:16:01.840
Matt Clark: You know, 100 meters from the sampling location, 200 meters, 300 meters, 400 meters, so different concentric circles of distance.

107
00:16:01.950 --> 00:16:15.519
Matt Clark: was considered. The imagery is 5 meter pixel size, so there's… as you get bigger and bigger circles, you're incorporating more and more, pixels. And we looked at the mean value of these engineered features, as well as their standard deviation.

108
00:16:15.660 --> 00:16:19.360
Matt Clark: And this thing called DIP, which gives you a sense of bimodality.

109
00:16:19.770 --> 00:16:25.930
Matt Clark: And we put all of those features, and there was a lot of them, into the random forest.

110
00:16:26.220 --> 00:16:33.690
Matt Clark: And we found that 350 meters was kind of the optimal scale of achieving our R-squared of .36, so…

111
00:16:33.920 --> 00:16:39.300
Matt Clark: That's, you know, it's not amazing, but for ecology, it's not so bad.

112
00:16:39.860 --> 00:16:44.239
Matt Clark: And this is birds, frogs, and insects that we're estimating here.

113
00:16:44.690 --> 00:16:54.699
Matt Clark: We did a little feature selection, a kind of down select, and we found, actually, with just 11 optimal features, we got… we did a better job.45R squared.

114
00:16:55.570 --> 00:17:03.350
Matt Clark: So that's kind of what, you know, that's the base, I guess, the benchmark, is trying to achieve something better than .45.

115
00:17:03.720 --> 00:17:11.490
Matt Clark: And we were able then to, go and make a map.

116
00:17:12.000 --> 00:17:21.630
Matt Clark: So over where they flew their hyperspectral sensor with that, that Learjet airplane, so it's not even the whole region, you know, but

117
00:17:21.710 --> 00:17:35.300
Matt Clark: They flew this all over different boxes, and then we were able to basically take these engineered features and our random forest model and make a map of estimated richness. So this is from, like, 10 species to 35 species.

118
00:17:35.530 --> 00:17:41.439
Matt Clark: And, so you can see there's variation across the landscape. Some areas are higher than others, and that's interesting.

119
00:17:41.550 --> 00:17:48.730
Matt Clark: For conservation, if you're trying to plant, say, protected areas, or, you know, where you want to conserve areas with maybe higher biodiversity.

120
00:17:48.950 --> 00:17:52.729
Matt Clark: Or just look at how biodiversity is changing over… over space.

121
00:17:54.070 --> 00:18:00.220
Matt Clark: And so that's where we're at in our research, and we're working on a paper on this right now.

122
00:18:00.430 --> 00:18:07.739
Matt Clark: But I'm interested in this, kind of a machine learning approach to get these features, rather than…

123
00:18:07.930 --> 00:18:21.900
Matt Clark: engineering the features, kind of what I call kind of boutique handcrafted features, like, let's just let the machine figure out, what are the important features. And in the past, we have used convolutional neural networks.

124
00:18:21.970 --> 00:18:31.069
Matt Clark: But these transformers are kind of interesting, because you can put the spatial, temporal, spectral aspect Together. Okay.

125
00:18:31.440 --> 00:18:38.469
Matt Clark: So I did… I haven't done a lot of deep research on this, but I did find a paper I think could be useful as a starting point.

126
00:18:39.050 --> 00:18:49.999
Matt Clark: And it's really just, like, a working conference paper. Here's the link to it here. But it's called Masked Vision Transformers. So Vision Transformers, you'll see, are kind of,

127
00:18:50.700 --> 00:18:52.410
Matt Clark: Talked about as VIT.

128
00:18:52.650 --> 00:18:57.279
Matt Clark: is the little acronym they use. So imagine you have, hyperspectral data.

129
00:18:57.520 --> 00:19:00.039
Matt Clark: Which they represent… they often represent that as a cube.

130
00:19:00.140 --> 00:19:12.240
Matt Clark: Where, the layer cake is basically… so you've got your space as the top layer, but as you go deeper, you're moving through spectral space, so from, like, visible through near-infrared through shortwave infrared.

131
00:19:12.700 --> 00:19:20.960
Matt Clark: And what they're doing is they're holding out, like, a whole block of the data, spectrally and, you know, spatially, like a cube.

132
00:19:21.410 --> 00:19:28.059
Matt Clark: And then, you know, doing your spectral spatial transformer.

133
00:19:28.160 --> 00:19:36.939
Matt Clark: With masking to try and reconstruct the hyperspectral cube, and in that process, learn some embeddings.

134
00:19:37.100 --> 00:19:41.679
Matt Clark: That can then be used for, in this case, they did classification, was their application.

135
00:19:42.820 --> 00:19:48.169
Matt Clark: So here's just another figure from the paper. It's actually, in these tokens.

136
00:19:48.540 --> 00:19:52.499
Matt Clark: Which represent, like, different spatial locations.

137
00:19:52.660 --> 00:19:55.149
Matt Clark: And spectral locations.

138
00:19:55.950 --> 00:20:06.699
Matt Clark: So you've got spectral trans… so imagine this going from, like, visible to… to, let's say, like, blue, green, red, near-infrared, you know, shortwave infrared, kind of…

139
00:20:07.240 --> 00:20:11.019
Matt Clark: Spaced maybe evenly across your spectral space.

140
00:20:11.350 --> 00:20:21.669
Matt Clark: So that'd be, like, a spectral token… tokenization, and this is more of a spatial tokenization. And that's what they're trying to show over here, is kind of spectral spatial.

141
00:20:22.020 --> 00:20:28.509
Matt Clark: And it's at those token locations is where the embeddings get, created.

142
00:20:29.480 --> 00:20:41.920
Matt Clark: Okay, that's as far… that's kind of, like, as much as I know at this point, but, there is a GitHub that could be used as a starting point with some example hyperspectral data sets.

143
00:20:41.930 --> 00:20:51.650
Matt Clark: I think that would probably be the most… the starting point, was just trying to, like, get their code working and see how it works on their hyperspectral data.

144
00:20:53.420 --> 00:20:59.200
Matt Clark: And then move on to using the Bioscape hyperspectral data.

145
00:21:00.380 --> 00:21:01.949
Matt Clark: So,

146
00:21:02.350 --> 00:21:09.450
Matt Clark: I invite you to go to our little story map here, that can tell you more about the Bioscape project.

147
00:21:09.540 --> 00:21:28.660
Matt Clark: That we're working on. Ours is called Biosoundscape. That's, like, our particular part. There's other people working on plant diversity and genetic diversity in different projects, but ours is sound, so we're called BioSoundscape. So there's a lot more, kind of, about the project and what, you know, some of the data sets are at this site.

148
00:21:28.750 --> 00:21:37.270
Matt Clark: But what I envision, I don't have a whole lot more than this, is, actually, let's just go to the GitHub. Let's do that.

149
00:22:04.430 --> 00:22:13.109
Matt Clark: Yeah, here it is. Okay, so at the GitHub for this paper, it's nice of them to provide that,

150
00:22:14.340 --> 00:22:21.710
Matt Clark: They do have some initial code for doing the pre-training, and that would,

151
00:22:22.270 --> 00:22:26.509
Matt Clark: basically create the embeddings, and they have some example data sets, they have…

152
00:22:26.830 --> 00:22:34.000
Matt Clark: some data that's NMAP, which is a, experimental satellite

153
00:22:34.520 --> 00:22:38.120
Matt Clark: Hyperspectral sensor with 30 meter pixel sizes.

154
00:22:38.590 --> 00:22:43.910
Matt Clark: That can be used. So, our data are gonna be,

155
00:22:44.690 --> 00:22:49.009
Matt Clark: So, it's not from space, it's from an airplane, and it's just in those flight boxes.

156
00:22:49.790 --> 00:22:52.329
Matt Clark: Our data are 5 meter, because it's from an airplane.

157
00:22:52.680 --> 00:22:54.360
Matt Clark: And…

158
00:22:54.520 --> 00:23:08.679
Matt Clark: But kind of similar, I think we'll have more… more parts of the spectrum measured. So, I think the adverse data are going to be about 430 bands, as we call them, different parts of the spectrum, where MMAP, I believe, is about 210.

159
00:23:08.780 --> 00:23:14.699
Matt Clark: bands or parts of the spectrum. So there's gonna be some different spectral resolution to deal with.

160
00:23:15.000 --> 00:23:17.790
Matt Clark: But what I want to do is,

161
00:23:18.360 --> 00:23:26.270
Matt Clark: Figure out how to get, our… hyperspectral data, Avarus NG,

162
00:23:26.480 --> 00:23:36.860
Matt Clark: Through a spectral spatial transformer, similar to this one, That will give us embeddings.

163
00:23:37.120 --> 00:23:44.550
Matt Clark: And then take those embeddings, And relate those to our species richness at our sites.

164
00:23:45.380 --> 00:23:51.390
Matt Clark: and estimate species richness. So basically, do what we did with our engineered features.

165
00:23:51.640 --> 00:24:00.230
Matt Clark: And it could be a random forest at that point, where you throw in these embedded features into a random forest, that'd be totally fine.

166
00:24:00.360 --> 00:24:03.829
Matt Clark: You could try support vector machining, that'd be another approach.

167
00:24:04.030 --> 00:24:12.790
Matt Clark: But maybe we do both. But anyways, the idea is to, estimate species richness.

168
00:24:13.030 --> 00:24:20.929
Matt Clark: And we're gonna do this in a way that's kind of smart, statistically, where we've got sites that are clustered.

169
00:24:22.160 --> 00:24:41.650
Matt Clark: That already, kind of, we know… I gave them cluster numbers, so things that are close in space are given a cluster number. And what we do is we… when we're estimating richness, we're going to want to do it in a leave-one-out approach, or leave one cluster-out approach. So you withhold one cluster, build the model on the other data clusters.

170
00:24:41.910 --> 00:24:57.500
Matt Clark: And then predict, you know, on that model, predict it back to the held-out cluster. That'll give you a less biased model, or estimate of accuracy, when you do it that way, a cross-validated accuracy assessment of richness.

171
00:24:59.160 --> 00:25:09.620
Matt Clark: than if you were just to throw everything all at once into the same model. So, and it lets you use all your data and have some independent data held out at each, fold.

172
00:25:09.860 --> 00:25:11.459
Matt Clark: And the cross-validation.

173
00:25:11.750 --> 00:25:16.630
Matt Clark: Okay, so that's kind of where it's going, that's what I want to do, that's the goal,

174
00:25:16.850 --> 00:25:35.349
Matt Clark: And I'm hoping Professor Gill can help move things along. And I think, again, I think the starting place is with this GitHub, and just trying to, like, maybe recreate, at least the step of running the spatial spectral transformer to get embeddings.

175
00:25:35.940 --> 00:25:42.850
Matt Clark: And once that's working, we can then… I can give you the hyperspectral data from Africa, and we can start using that.

176
00:25:43.150 --> 00:25:46.690
Matt Clark: To, you know, run through the code and get the embeddings.

177
00:25:47.310 --> 00:25:49.900
Matt Clark: Okay, I'll stop there. Bye.

