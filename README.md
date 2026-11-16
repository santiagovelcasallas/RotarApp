# RotarApp

RotarApp is a decision-support tool for **agroecological crop rotation** (no agrochemicals and minimal artificial irrigation) in the **lowland area of Valle del Cauca, Colombia**. It combines NASA Earth observations, local soil information, crop characteristics, and farmers' priorities.

NASA Space Apps Challenge: *Crop Rotation Decision-Support Tool*.

> RotarApp **supports, but does not replace**, farmers' judgment or local agronomic advice. Each result reports its uncertainty and identifies what should be validated in the field.

## What It Does

For each pixel in the study area, RotarApp compares **at least three crop rotations, including the current rotation**. Each rotation receives four scores:

| Score | Question |
|---|---|
| Water and climate | How much water deficit does it face, and how much irrigation would it require? |
| Erosion | How much soil is lost (RUSLE)? |
| Profitability | What profit is expected in a typical year and in a poor year? |
| Biological soil health | How does the soil microbiome change over time (estimated index)? |

For each strategy, the app explains **why it is recommended, what evidence supports it, what benefits it offers, where information gaps remain, and which risks require local review**.

## How It Works

```text
NASA and local data ──► 1. Pixel-level constraint filter (slope, drainage, climate, soil)
                              │  → viable crops for each pixel
                              ▼
                         2. Pixel-level simulator (cellular automaton, ~10-day steps)
                              state: soil water · vigor (NDVI) · disease inoculum · soil health
                              run across many climate years (Monte Carlo by ENSO phase)
                              │
                              ├─► 3. Erosion (RUSLE)
                              ├─► 4. Profitability (NDVI → yield × price − costs)
                              └─► 5. Soil microbiome index
                              ▼
                         6. Crop rotation comparator, weighted by farmer priorities
                              ▼
                         Precomputed results (JSON / GeoTIFF) ──► App (Lovable)
```

For details, see the [architecture](docs/arquitectura.md), [datasets](docs/datos.md), and [candidate rotations](docs/rotaciones.md) documentation.

## Repository Structure

```text
RotarApp/
├── config/              # Study area and crop parameters
├── data/                # Data (raw/ and interim/ are not version-controlled)
├── docs/                # Architecture, datasets, and rotations
├── notebooks/           # Exploration and prototypes
├── src/rotarapp/
│   ├── datos/           # Data download and harmonization to a common grid
│   ├── filtro/          # 1. Pixel-level constraints
│   ├── simulador/       # 2. Cellular automaton (water, vigor, disease, soil)
│   ├── erosion/         # 3. RUSLE
│   ├── economia/        # 4. NDVI → yield → profit
│   ├── microbiota/      # 5. Biological soil health index
│   └── comparador/      # 6. Scores, weighting, and explanations
├── app/                 # Data contract with the Lovable app
└── tests/
```

## References

### Soil Health and Microbiome Index

- Bünemann, E. K., Bongiorno, G., Bai, Z., Creamer, R. E., De Deyn, G., de Goede, R., Fleskens, L., Geissen, V., Kuyper, T. W., Mäder, P., Pulleman, M., Sukkel, W., van Groenigen, J. W., & Brussaard, L. (2018). Soil quality – A critical review. *Soil Biology and Biochemistry, 120*, 105–125. https://doi.org/10.1016/j.soilbio.2018.01.030
- Delgado-Baquerizo, M., Maestre, F. T., Reich, P. B., Jeffries, T. C., Gaitan, J. J., Encinar, D., Berdugo, M., Campbell, C. D., & Singh, B. K. (2016). Microbial diversity drives multifunctionality in terrestrial ecosystems. *Nature Communications, 7*, Article 10541. https://doi.org/10.1038/ncomms10541
- Liu, Q., Zhao, Y., Li, T., Chen, L., Chen, Y., & Sui, P. (2023). Changes in soil microbial biomass, diversity, and activity with crop rotation in cropping systems: A global synthesis. *Applied Soil Ecology, 186*, Article 104815. https://doi.org/10.1016/j.apsoil.2023.104815
- McDaniel, M. D., Tiemann, L. K., & Grandy, A. S. (2014). Does agricultural crop diversity enhance soil microbial biomass and organic matter dynamics? A meta-analysis. *Ecological Applications, 24*(3), 560–570. https://doi.org/10.1890/13-0616.1
- Wagg, C., Bender, S. F., Widmer, F., & van der Heijden, M. G. A. (2014). Soil biodiversity and soil community composition determine ecosystem multifunctionality. *Proceedings of the National Academy of Sciences, 111*(14), 5266–5270. https://doi.org/10.1073/pnas.1320054111

### Local Crop Pests and Diseases

The source IDs (S01, S03, etc.) are retained as citation identifiers.

#### Sugarcane

- **S01:** “Spatiotemporal distribution of *Diatraea* spp. in the Cauca River Valley” (original title: “Distribución espacio-temporal de *Diatraea* spp. en el valle del río Cauca”). *Revista Colombiana de Entomología* (2018).
- **S03:** Colombian Agricultural Institute (ICA). Resolution 1932 of 2007 (phytosanitary emergency related to spittlebugs in Valle del Cauca).

#### Rice

- **S09:** UPRA. Rice production cost profile for southern Valle del Cauca (2025).

#### Maize, Sorghum, and Soybean

- **S13:** ICA Valle Sectional Office. Resolution 00016286 of 2026.
- **S14:** ICA Valle Sectional Office. Resolution 00009597 of 2023.
- **S15:** ICA Valle Sectional Office. Resolution 889 of 2022.
- **S16:** Management of *Dalbulus maidis* in Valle del Cauca.
- **S19:** “Damage caused by *Diatraea* in maize and sorghum in Valle del Cauca” (original title: “Daño de *Diatraea* en maíz y sorgo en el Valle del Cauca”; older study). *Acta Agronómica*.
- **S20:** ICA. Resolution 106 of 2004 (soybean rust alert for Valle del Cauca and Cauca).

#### Beans and Vegetables

- **S24:** “Bean leaf crumple virus and *Bemisia tabaci* in Valle del Cauca” (original title: “Virus del arrugamiento foliar del fríjol y *Bemisia tabaci* en el Valle”). *Revista Colombiana de Entomología*.
- **S27:** “Biotype B of *Bemisia tabaci*, a pepper pest in Valle del Cauca” (original title: “Biotipo B de *Bemisia tabaci*, plaga de pimentón en el Valle del Cauca”). *Revista Colombiana de Entomología*.
- **S66:** “Bean diseases in Valle del Cauca” (original title: “Enfermedades del fríjol en el Valle del Cauca”; older study). *Acta Agronómica* (1958).

#### Coffee

- **S33:** Cenicafé. Phytosanitary Bulletin 20 (2026).
- **S34:** Cenicafé. Phytosanitary Bulletin 18 (2026).
- **S36:** Cenicafé. Phytosanitary Bulletin 17 (2025).

#### Plantain

- **S38:** Universidad del Pacífico. Compost and plantain health in Buenaventura.
- **S39:** ICA (2014). Fusarium R4T surveillance (farms in Valle del Cauca).
- **S41:** “Plantain disease in Valle del Cauca” (original title: “Enfermedad del plátano en el Valle del Cauca”; older study). *Acta Agronómica* (1968).

#### Fruit Crops

- **S46:** ICA (2014). Surveillance of quarantine pests affecting avocado in Valle del Cauca.
- **S50:** Corpoica. Leafroller affecting guava variety ICA-1 in northern Valle del Cauca.
- **S52:** *Dynamis borassi* in Buenaventura. *Revista Colombiana de Entomología, 46*(2) (2020).
- **S53:** Pardo-Locarno et al. (2014). Phytosanitary diagnosis of peach palm in Buenaventura.
- **S54:** “Peach palm in Valle del Cauca and Chocó” (original title: “Chontaduro en el Valle y Chocó”; older study). *Acta Agronómica*.
- **S55:** “Borojo, a promising crop” (original title: “El borojó, cultivo promisorio”; older study). *Acta Agronómica*.
- **S58:** Portal Frutícola. Universidad Nacional de Colombia, Palmira: phytosanitary diagnosis of 13 fruit crops in Valle del Cauca.

### Crop Rotation

- Young, M. D., Ros, G. H., & de Vries, W. (2021). Impacts of agronomic measures on crop, soil, and environmental indicators: A review and synthesis of meta-analysis. *Agriculture, Ecosystems & Environment, 319*, 107551. https://doi.org/10.1016/j.agee.2021.107551
- Peralta, A. L., Sun, Y., McDaniel, M. D., & Lennon, J. T. (2018). Crop rotational diversity increases disease suppressive capacity of soil microbiomes. *Ecosphere, 9*(5), e02235. https://doi.org/10.1002/ecs2.2235
- King, A. E., & Blesh, J. (2018). Crop rotations for increased soil carbon: Perenniality as a guiding principle. *Ecological Applications, 28*(1), 249–261. https://doi.org/10.1002/eap.1648

### Additional Reference

- Hadi et al. “Land degradation is associated with larger crop yield gaps across global croplands.” *Nature Food*. https://doi.org/10.1038/s43016-026-01382-5. [University of Bonn article](https://www.uni-bonn.de/en/news/144-2026).

## Team

- Nicol Fandiño Quiceno
- Santiago Velez Casallas
- Samuel Pavas Cadavid
- Luis Felipe Yanez
- Luisa Polanco
