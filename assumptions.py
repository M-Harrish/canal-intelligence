"""Economic parameters, each tagged with its provenance.

Every number used in the budget allocator (step 6) and the failure simulation
(step 7) lives here. Each carries a `status`:

  SOURCED  traceable to a named public source (cited in `source`)
  DERIVED  arithmetic on a SOURCED figure (the arithmetic is shown)
  ASSUMED  a placeholder — MUST be replaced before any real decision

The UI and CLI print these tags next to every rupee figure, so nothing that
reaches a decision-maker is silently invented.
"""


class Param:
    def __init__(self, value, unit, status, source, note=""):
        self.value = value
        self.unit = unit
        self.status = status
        self.source = source
        self.note = note

    def __float__(self):
        return float(self.value)

    def label(self):
        tag = {
            "SOURCED": "sourced",
            "DERIVED": "derived",
            "ASSUMED": "ASSUMED VALUE — replace with PWD schedule of rates",
        }[self.status]
        return f"{self.value:,.0f} {self.unit} [{tag}]"

    def describe(self):
        out = f"{self.value:,.2f} {self.unit}  ({self.status})\n    source: {self.source}"
        if self.note:
            out += f"\n    note:   {self.note}"
        return out


# --------------------------------------------------------------- desilting cost
DESILT_COST_PER_KM = Param(
    value=239_644,
    unit="INR/km",
    status="DERIVED",
    source=(
        "Deccan Chronicle, 18 Aug 2019, 'Water released from Grand Anicut': "
        "'desilting to a distance of 2211.63 kms in rivers and canals has been "
        "taken up in the three districts at a cost of Rs 53 crore' "
        "(Thanjavur, Tiruvarur, Nagapattinam; PWD 2019-20). "
        "https://www.deccanchronicle.com/nation/current-affairs/180819/"
        "water-released-from-grand-anicut.html"
    ),
    note=(
        "Rs 53,00,00,000 / 2211.63 km = Rs 2,39,644/km. This is an AVERAGE over "
        "a whole programme in exactly our three districts, not a schedule rate. "
        "Real cost varies with section, silt depth and disposal distance, and "
        "the 2019-20 figure is not inflation-adjusted. Replace with the PWD "
        "schedule of rates before any actual tendering decision."
    ),
)

# Larger sections cost more per km to desilt than field channels. No public
# per-order breakdown was found, so these multipliers are openly assumed.
COST_MULTIPLIER_BY_TYPE = {
    "Main Canal": 2.0,
    "Branch Canal": 1.5,
    "Distributary": 1.0,
    "Minor": 0.7,
    "Sub Minor": 0.5,
    "Sub Sub Minor": 0.4,
    "Water Course": 0.4,
    "Feeder": 1.5,
}
COST_MULTIPLIER_STATUS = Param(
    value=0, unit="multiplier", status="ASSUMED",
    source="No public per-canal-order desilting rate breakdown located.",
    note=("Cost multipliers by canal order are ASSUMED VALUES — replace with "
          "PWD schedule of rates. Set all to 1.0 to disable."),
)

# --------------------------------------------------------------- crop economics
PADDY_MSP_PER_QUINTAL = Param(
    value=2_369,
    unit="INR/quintal",
    status="SOURCED",
    source=(
        "CCEA, MSP for Kharif Crops, Marketing Season 2025-26 — paddy (common) "
        "Rs 2,369/quintal (Grade A Rs 2,389). PIB release 2131983. "
        "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2131983"
    ),
)

PADDY_YIELD_T_PER_HA = Param(
    value=3.45,
    unit="tonnes paddy/ha",
    status="DERIVED",
    source=(
        "Tamil Nadu rice yield 2023-24 = 2.31 t/ha (The Hindu / TN Dept of "
        "Economics & Statistics, vs all-India 2.74 t/ha). Conversion norm: "
        "100 kg paddy -> 67 kg raw rice."
    ),
    note=(
        "2.31 t/ha milled rice / 0.67 = 3.45 t/ha paddy. MSP is paid on PADDY, "
        "so paddy yield is the right basis. State-average, not delta-specific; "
        "delta yields are typically higher."
    ),
)

CROPPING_INTENSITY = Param(
    value=1.0,
    unit="seasons/year",
    status="ASSUMED",
    source="Set to 1.0 (single season) deliberately.",
    note=(
        "The Cauvery delta commonly grows Kuruvai + Samba (up to 2 seasons) "
        "where water allows. Left at 1.0 so revenue figures are CONSERVATIVE "
        "and clearly a lower bound. Raise only with district crop statistics."
    ),
)


def gross_revenue_per_ha():
    """INR per hectare per year of paddy, at MSP."""
    price_per_tonne = PADDY_MSP_PER_QUINTAL.value * 10  # 1 t = 10 quintal
    return (price_per_tonne * PADDY_YIELD_T_PER_HA.value
            * CROPPING_INTENSITY.value)


def cost_to_desilt(length_m, can_type):
    mult = COST_MULTIPLIER_BY_TYPE.get(can_type, 1.0)
    return (length_m / 1000.0) * DESILT_COST_PER_KM.value * mult


ALL_PARAMS = {
    "DESILT_COST_PER_KM": DESILT_COST_PER_KM,
    "COST_MULTIPLIER_BY_TYPE": COST_MULTIPLIER_STATUS,
    "PADDY_MSP_PER_QUINTAL": PADDY_MSP_PER_QUINTAL,
    "PADDY_YIELD_T_PER_HA": PADDY_YIELD_T_PER_HA,
    "CROPPING_INTENSITY": CROPPING_INTENSITY,
}


def print_all():
    print("=" * 72)
    print("ECONOMIC PARAMETERS")
    print("=" * 72)
    for name, p in ALL_PARAMS.items():
        print(f"\n{name}\n    {p.describe()}")
    print(f"\nGROSS REVENUE PER HA (derived): "
          f"Rs {gross_revenue_per_ha():,.0f}/ha/year")
    print("    = MSP Rs {:,}/quintal x 10 x {} t/ha x {} season(s)".format(
        PADDY_MSP_PER_QUINTAL.value, PADDY_YIELD_T_PER_HA.value,
        CROPPING_INTENSITY.value))
    assumed = [n for n, p in ALL_PARAMS.items() if p.status == "ASSUMED"]
    if assumed:
        print(f"\n!! {len(assumed)} ASSUMED parameter(s): {', '.join(assumed)}")
        print("   Replace with PWD schedule of rates before operational use.")
    print("=" * 72)


if __name__ == "__main__":
    print_all()
