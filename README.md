## Antarctic Plateau Salt Relative Humidity and Temperature (apsrht) Timeseries Data

 
### Description:
<div align="justify">
This project investigates a time series logged by a sensor installation on the Antarctic Plateau. The device began logging on 12/19/2018 at 0:00 hours and recorded its last measurement 02/02/2019 at 23:45 hours, at intervals of 15 minutes. During those few months, every 15 minutes, the sensors took a measurement of the temperature and relative humidity of a 300g pure NaCl sample, hence "salt". For more information about the data logger setup/hardware see the Data Source section below. Temperature is in degrees Celsius and relative humidity is a percentage. Corrected relative humidity is the calculation with respect to ice, and is the variable I will use, as the temperature never exceeds 0 degree Celsius. The point of this project is to explore the relationship between temperature and relative humidity occurring at a certain geographic point in Antarctica. I will use the data to make predictions for future temperatures in this area, which I will then compare to actual observed temperatures that were recorded in the following years after this data logger recorded these values. This project also serves as an analogue for some of my future studies pertaining to the cryosphere.
</div>


### Location Data Was Collected:
Antarctic Plateau
[ -77.0730555556, 101.4983333333 ]


### Data Source:

* Astrobiology Habitable Environments Database (AHED) - NASA [DOI](https://doi.org/10.48667/sy0j-yc84)

* Citation: Davila, A; Parro, V. Antarctic Plateau-Salt Relative Humidity and Temperature. Astrobiology Habitable Environment Database. Retrieved Date: Jun 23, 2026. https://doi.org/10.48667/sy0j-yc84

<div align="justify">
"This dataset includes relative humidity and temperature values measured inside a NaCl sample on the Antarctic Plateau. Approximately 300 g of pure NaCl were placed in a squared plastic container with RH and T sensors and a data logger (U23 Pro V2 from HOBO) inside the salt set to record data every 15 minutes. The box was sealed, and holes were drilled on the top of the plastic container for air circulation. RH and T data from the salt were obtained daily during the 2,538 km transect from the Novolazarevskaya station to Dome Fuji (altitude: 3,810 m), and back to the station. The data were downloaded at the end of the transect and RH values were corrected for sub-zero temperatures. Briefly, RH sensors record humidity with respect to liquid water, even at subfreezing temperatures, and often have offsets at high humidity and low temperature that need to be considered. To deal with this issue (i.e. sub-zero temperatures) on the Antarctic Plateau, the RH readings were corrected using the following empirical formula: RHice = RH − 2 − 0.65T, where RH is in percent, T is the temperature in Celsius, and RHice is the RH with respect to ice, and is the final corrected value."
</div>

