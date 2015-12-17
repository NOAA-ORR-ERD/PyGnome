'''
    This is where we handle the initialization of the estimated oil properties.
    This will be the 'real' oil record that we use.

    Basically, we have an Estimated object that is a one-to-one relationship
    with the Oil object.  This is where we will place the estimated oil
    properties.
'''
from math import log, log10, exp, fabs
import transaction

import numpy as np

from oil_library.models import (ImportedRecord, Oil, Estimated,
                                Density, KVis, Cut,
                                SARAFraction, SARADensity,
                                MolecularWeight)

from oil_library.utilities import (get_boiling_points_from_api,
                                   get_viscosity)


class OilRejected(Exception):
    '''
        Custom exception for Oil initialization that we can raise if we
        decide we need to reject an oil record for any reason.
    '''
    def __init__(self, message, oil_name, *args):
        # without this you may get DeprecationWarning
        self.message = message

        # Special attribute you desire with your Error,
        # perhaps the value that caused the error?:
        self.oil_name = oil_name

        # allow users initialize misc. arguments as any other builtin Error
        super(OilRejected, self).__init__(message, oil_name, *args)

    def __repr__(self):
        return '{0}(oil={1}, errors={2})'.format(self.__class__.__name__,
                                                 self.oil_name,
                                                 self.message)


def process_oils(session_class):
    session = session_class()
    record_ids = [r.adios_oil_id for r in session.query(ImportedRecord)]
    session.close()

    print '\nAdding Oil objects...'
    for record_id in record_ids:
        # Note: committing our transaction for every record slows the
        #       import job significantly.  But this is necessary if we
        #       want the option of rejecting oil records.
        session = session_class()
        transaction.begin()
        rec = (session.query(ImportedRecord)
               .filter(ImportedRecord.adios_oil_id == record_id)
               .one())

        try:
            add_oil(rec)
            transaction.commit()
        except OilRejected as e:
            print repr(e)
            transaction.abort()


def add_oil(record):
    print 'Estimations for {0}'.format(record.adios_oil_id)
    oil = Oil()
    oil.estimated = Estimated()

    add_demographics(record, oil)
    add_densities(record, oil)
    add_viscosities(record, oil)
    add_oil_water_interfacial_tension(record, oil)
    add_oil_seawater_interfacial_tension(record, oil)

    add_pour_point(record, oil)
    add_flash_point(record, oil)
    add_emulsion_water_fraction_max(record, oil)

    add_metals(record, oil)
    add_imported_sara_fractions(record, oil)
    add_misc_fractions(record, oil)

    add_resin_fractions(record, oil)
    add_asphaltene_fractions(record, oil)

    add_bullwinkle_fractions(record, oil)
    add_adhesion(record, oil)
    add_sulphur_mass_fraction(record, oil)
    add_soluability(record, oil)
    add_distillation_cut_boiling_point(record, oil)
    add_molecular_weights(record, oil)
    add_component_densities(record, oil)
    add_saturate_aromatic_fractions(record, oil)
    adjust_molecular_weights(oil)
    adjust_resin_asphaltene_fractions(record, oil)

    add_k0y(record, oil)

    reject_oil_if_bad(record, oil)

    record.oil = oil


def add_demographics(imported_rec, oil):
    oil.name = imported_rec.oil_name
    oil.adios_oil_id = imported_rec.adios_oil_id


def add_densities(imported_rec, oil):
    '''
        Rules:
        - If no density value exists, estimate it from the API.
          So at the end, we will always have at least one density at
          15 degrees Celsius.
        - If a density measurement at some temperature exists, but no API,
          then we estimate API from density.
          So at the end, we will always have an API value.
        - In both the previous cases, we have estimated the corollary values
          and ensured that they are consistent.  But if a record contains both
          an API and a number of densities, these values may conflict.
          In this case, we will reject the creation of the oil record.
        - This is not in the document, but Bill & Chris have verbally
          stated they would like there to always be a 15C density value.
    '''
    for d in imported_rec.densities:
        if d.kg_m_3 is not None:
            oil.densities.append(d)

    if imported_rec.api is not None:
        oil.api = imported_rec.api
    elif oil.densities:
        # estimate our api from density
        d_0 = density_at_temperature(oil, 273.15 + 15)

        oil.api = (141.5 * 1000 / d_0) - 131.5
        # oil.estimated.api = True
    else:
        print ('Warning: no densities and no api for record {0}'
               .format(imported_rec.adios_oil_id))

    if not [d for d in oil.densities
            if (d.ref_temp_k is not None and
                np.isclose(d.ref_temp_k, 273.0 + 15, atol=.15))]:
        # add a 15C density from api
        kg_m_3, ref_temp_k = estimate_density_from_api(oil.api)

        oil.densities.append(Density(kg_m_3=kg_m_3,
                                     ref_temp_k=ref_temp_k,
                                     weathering=0.0))
        # oil.estimated.densities = True


def estimate_density_from_api(api):
    kg_m_3 = 141.5 / (131.5 + api) * 1000
    ref_temp_k = 273.15 + 15

    return kg_m_3, ref_temp_k


def density_at_temperature(oil_rec, temperature, weathering=0.0):
    # first, get the density record closest to our temperature
    density_list = [(d, abs(d.ref_temp_k - temperature))
                    for d in oil_rec.densities
                    if (d.ref_temp_k is not None and
                        d.weathering == weathering)]

    if density_list:
        density_rec = sorted(density_list, key=lambda d: d[1])[0][0]
        d_ref = density_rec.kg_m_3
        t_ref = density_rec.ref_temp_k
    else:
        if oil_rec.api is None:
            # We have no densities at our requested weathering, and no api
            # We cannot make a computation.
            return None
        else:
            d_ref, t_ref = estimate_density_from_api(oil_rec.api)

    k_pt = 0.008
    if density_list and fabs(t_ref - temperature) > (1 / k_pt):
        # even if we got some measured densities, they could be at
        # temperatures that is out of range for our algorithm.
        return None

    return d_ref / (1 - k_pt * (t_ref - temperature))


def add_viscosities(imported_rec, oil):
        '''
            Get a list of all kinematic viscosities associated with this
            oil object.  The list is compiled from the stored kinematic
            and dynamic viscosities associated with the oil record.
            The viscosity fields contain:
              - kinematic viscosity in m^2/sec
              - reference temperature in degrees kelvin
              - weathering ???
            Viscosity entries are ordered by (weathering, temperature)
            If we are using dynamic viscosities, we calculate the
            kinematic viscosity from the density that is closest
            to the respective reference temperature
        '''
        kvis, estimated = get_kvis(imported_rec)

        kvis.sort(key=lambda x: (x[2], x[1]))
        kwargs = ['m_2_s', 'ref_temp_k', 'weathering']

        for v in kvis:
            oil.kvis.append(KVis(**dict(zip(kwargs, v))))

        if any(estimated):
            oil.estimated.viscosities = True


def get_kvis(imported_rec):
    if imported_rec.kvis is not None:
        viscosities = [(k.m_2_s,
                        k.ref_temp_k,
                        (0.0 if k.weathering is None else k.weathering))
                       for k in imported_rec.kvis
                       if k.ref_temp_k is not None]
    else:
        viscosities = []
    estimated = [False] * len(viscosities)

    for kv, t, w in get_kvis_from_dvis(imported_rec):
        if kvis_exists_at_temp_and_weathering(viscosities, t, w):
            continue

        viscosities.append((kv, t, w))
        estimated.append(True)

    return viscosities, estimated


def get_kvis_from_dvis(oil_rec):
    '''
        If we have any DVis records, we convert them to kinematic and return
        them.
        DVis records are correlated with a ref_temperature, and weathering.
        In order to convert dynamic viscosity to kinematic, we need to get
        the density at our reference temperature and weathering
    '''
    kvis_out = []

    if oil_rec.dvis:
        for dv, t, w in [(d.kg_ms,
                         d.ref_temp_k,
                         (0.0 if d.weathering is None else d.weathering))
                         for d in oil_rec.dvis
                         if d.kg_ms > 0.0 and d.ref_temp_k is not None]:
            density = density_at_temperature(oil_rec, t, w)

            # kvis = dvis/density
            if density is not None:
                kvis_out.append(((dv / density), t, w))

    return kvis_out


def kvis_exists_at_temp_and_weathering(kvis, temperature, weathering):
    return len([v for v in kvis
                if (v[1] == temperature and
                    v[2] == weathering)]) > 0


def add_oil_water_interfacial_tension(imported_rec, oil):
    if imported_rec.oil_water_interfacial_tension_n_m is not None:
        oil.oil_water_interfacial_tension_n_m = \
            imported_rec.oil_water_interfacial_tension_n_m
        oil.oil_water_interfacial_tension_ref_temp_k = \
            imported_rec.oil_water_interfacial_tension_ref_temp_k
    else:
        # estimate values from api
        if imported_rec.api is not None:
            api = imported_rec.api
        elif oil.api is not None:
            api = oil.api
        else:
            api = None

        oil.oil_water_interfacial_tension_n_m = (0.001 * (39 - 0.2571 * api))
        oil.oil_water_interfacial_tension_ref_temp_k = 273.15 + 15.0

        oil.estimated.oil_water_interfacial_tension_n_m = True
        oil.estimated.oil_water_interfacial_tension_ref_temp_k = True


def add_oil_seawater_interfacial_tension(imported_rec, oil):
    if imported_rec.oil_seawater_interfacial_tension_n_m is not None:
        oil.oil_seawater_interfacial_tension_n_m = \
            imported_rec.oil_seawater_interfacial_tension_n_m
        oil.oil_seawater_interfacial_tension_ref_temp_k = \
            imported_rec.oil_seawater_interfacial_tension_ref_temp_k


def add_pour_point(imported_rec, oil):
    '''
        If we already have pour point min-max values in our imported
        record, then we are good.  We simply copy them over.
        If we don't have them, then we will need to approximate them.

        If we have measured molecular weights for the distillation fractions
        then:
            (A) If molecular weight M_w in kg/kmol and mass fractions are
                given for all the oil fractions (j = 1...jMAX), than an
                average molecular weight for the whole oil can be estimated
                as:
                    M_w_avg = sum[1,jMAX](M_w(j) * fmass_j)
                    where fmass_j = mass fraction of component j.
                    (Note: jMAX = 2(N + 1) wherer N = number of
                           distillation cuts.  We sum over all the SARA
                           fractions but resins and asphaltenes do not
                           have distillation cut data.)
                Define SG = P_oil / 1000 kg as specific gravity.
                Define T_api = 311.15K = reference temperature for the oil
                                         kinematic viscosity.
                T_pp = (130.47 * SG^2.97) * \
                       M_w_avg^(0.61235 - 0.47357 * SG) * \
                       V_oil^(0.31 - 0.3283 * SG) * \
                       T_api
        else:
            (B) Pour point is estimated by reversing the viscosity-to-
                temperature correction in AIDOS2 and assuming that, at the
                pour point, viscosity is equal to 1 million centistokes.
    '''
    if (imported_rec.pour_point_min_k is not None or
            imported_rec.pour_point_max_k is not None):
        # we have values to copy over
        oil.pour_point_min_k = imported_rec.pour_point_min_k
        oil.pour_point_max_k = imported_rec.pour_point_max_k
    else:
        oil.pour_point_min_k = None
        if 0:
            # TODO: When would we have molecular weights?
            # if we have measured molecular weights for the
            # distillation fractions, then we use method 'A'
            # oil.pour_point_max_k = \
            #     estimate_pp_by_molecular_weights(imported_rec)
            pass
        else:
            oil.pour_point_max_k = estimate_pp_by_viscosity_ref(imported_rec)

        oil.estimated.pour_point_min_k = True
        oil.estimated.pour_point_max_k = True


def estimate_pp_by_viscosity_ref(imported_rec):
    # Get the viscosity measured at the lowest reference temperature
    kvis_rec = sorted(get_kvis(imported_rec)[0],
                      key=lambda x: (x[2], x[1]))[0]

    v_ref, t_ref = kvis_rec[0], kvis_rec[1]
    c_v1 = 5000.0

    return (c_v1 * t_ref) / (c_v1 - t_ref * log(v_ref))


def add_flash_point(imported_rec, oil):
    '''
        If we already have flash point min-max values in our imported
        record, then we are good.  We simply copy them over.
        If we don't have them, then we will need to approximate them.

        If we have measured distillation cut data
        then:
            (A) T_cut1 = the boiling point of the first pseudo-component cut.
                T_flsh = 117 + 0.69 * T_cut1
        else:
            (B) T_flsh = 457 - 3.34 * api
    '''
    if (imported_rec.flash_point_min_k is not None or
            imported_rec.flash_point_max_k is not None):
        # we have values to copy over
        oil.flash_point_min_k = imported_rec.flash_point_min_k
        oil.flash_point_max_k = imported_rec.flash_point_max_k
    else:
        oil.flash_point_min_k = None
        if len(imported_rec.cuts) > 0:
            # if we have measured distillation cuts, then we use method 'A'
            oil.flash_point_max_k = estimate_fp_by_cut(imported_rec)
        else:
            # we use method 'B'
            oil.flash_point_max_k = estimate_fp_by_api(oil)

        oil.estimated.flash_point_min_k = True
        oil.estimated.flash_point_max_k = True


def estimate_fp_by_cut(imported_rec):
    '''
        If we have measured distillation cut data:
            (A) T_cut1 = the boiling point of the first pseudo-component cut.
                T_flsh = 117 + 0.69 * T_cut1
    '''
    temp_cut_1 = sorted(imported_rec.cuts,
                        key=lambda x: x.vapor_temp_k)[0].vapor_temp_k

    return 117.0 + 0.69 * temp_cut_1


def estimate_fp_by_api(imported_rec):
    '''
        If we do *not* have measured distillation cut data, then use api:
            (B) T_flsh = 457 - 3.34 * api
    '''
    return 457.0 - 3.34 * imported_rec.api


def add_emulsion_water_fraction_max(imported_rec, oil):
    '''
        This quantity will be set after the emulsification approach in ADIOS3
        is finalized.  It will vary depending upon the emulsion stability.
        For now set f_w_max = 0.9 for crude oils and f_w_max = 0.0 for
        refined products.
    '''
    if imported_rec.product_type == 'Crude':
        oil.emulsion_water_fraction_max = 0.9
    elif imported_rec.product_type == 'Refined':
        oil.emulsion_water_fraction_max = 0.0

    oil.estimated.emulsion_water_fraction_max = True


def add_metals(record, oil):
    oil.nickel_ppm = record.nickel
    oil.vanadium_ppm = record.vanadium


def add_misc_fractions(record, oil):
    oil.polars_fraction = record.polars
    oil.benzene_fraction = record.benzene
    oil.sulphur_fraction = record.sulphur
    oil.paraffins_fraction = record.paraffins
    oil.wax_content_fraction = record.wax_content


def add_imported_sara_fractions(record, oil):
    oil.saturates_fraction = record.saturates
    oil.aromatics_fraction = record.aromatics
    oil.resins_fraction = record.resins
    oil.asphaltenes_fraction = record.asphaltenes


def add_resin_fractions(imported_rec, oil):
    try:
        if (imported_rec is not None and
            imported_rec.resins is not None and
                imported_rec.resins >= 0.0 and
                imported_rec.resins <= 1.0):
            f_res = imported_rec.resins
        else:
            a, b = get_corrected_density_and_viscosity(oil)

            f_res = (3.3 * a + 0.087 * b - 74.0)
            f_res /= 100.0  # percent to fractional value

            f_res = 0.0 if f_res < 0.0 else f_res

        oil.sara_fractions.append(SARAFraction(sara_type='Resins',
                                               fraction=f_res,
                                               ref_temp_k=1015.0))
    except:
        print 'Failed to add Resin fraction!'


def add_asphaltene_fractions(imported_rec, oil):
    try:
        if (imported_rec is not None and
            imported_rec.asphaltenes is not None and
                imported_rec.asphaltenes >= 0.0 and
                imported_rec.asphaltenes <= 1.0):
            f_asph = imported_rec.asphaltenes
        else:
            a, b = get_corrected_density_and_viscosity(oil)

            f_asph = (0.0014 * (a ** 3.0) +
                      0.0004 * (b ** 2.0) -
                      18.0)
            f_asph /= 100.0  # percent to fractional value

            f_asph = 0.0 if f_asph < 0.0 else f_asph

        oil.sara_fractions.append(SARAFraction(sara_type='Asphaltenes',
                                               fraction=f_asph,
                                               ref_temp_k=1015.0))
    except:
        print 'Failed to add Asphaltene fraction!'


def get_corrected_density_and_viscosity(oil):
    '''
        Get coefficients for calculating resin (and asphaltene) fractions
        based on Merv Fingas' empirical analysis of ESTC oil properties
        database.
        - Bill has clarified that we want to get the coefficients for just
          the 15C Density
        - Mervs calculations depend on a density measured in g/mL and a
          viscosity measured in mPa.s, so we do a conversion here.
    '''
    try:
        temperature = 273.15 + 15
        P0_oil = density_at_temperature(oil, temperature)
        V0_oil = get_viscosity(oil, temperature)
        a = 10 * exp(0.001 * P0_oil)
        b = 10 * log(1000.0 * P0_oil * V0_oil)

    except:
        print 'get_corrected_density_and_viscosity() generated exception:'
        print '\toil = ', oil
        print '\toil.kvis = ', oil.kvis
        print '\tP0_oil = ', density_at_temperature(oil, temperature)
        print '\tV0_oil = ', get_viscosity(oil, temperature)
        raise

    return a, b


def add_bullwinkle_fractions(imported_rec, oil):
    '''
        This is the mass fraction that must evaporate or dissolve before
        stable emulsification can begin.
        - For this estimation, we depend on an oil object with a valid
          asphaltene fraction or a valid api
        - This is a scalar value calculated with a reference temperature of 15C
        - For right now we are referencing the Adios2 code file
          OilInitialize.cpp, function CAdiosData::Bullwinkle(void)
    '''
    if imported_rec.product_type == "refined":
        bullwinkle_fraction = 1.0
    elif imported_rec.emuls_constant_max is not None:
        bullwinkle_fraction = imported_rec.emuls_constant_max
    else:
        # product type is crude
        Ni = (imported_rec.nickel
              if imported_rec.nickel is not None else 0.0)
        Va = (imported_rec.vanadium
              if imported_rec.vanadium is not None else 0.0)
        f_asph = [af.fraction
                  for af in oil.sara_fractions
                  if (af.sara_type == 'Asphaltenes' and
                      af.fraction > 0)]
        f_asph = f_asph[0] if len(f_asph) > 0 else 0.0

        if (Ni > 0.0 and Va > 0.0 and Ni + Va > 15.0):
            bullwinkle_fraction = 0.0
        elif f_asph > 0.0:
            # Bullvalue = 0.32 - 3.59 * f_Asph
            bullwinkle_fraction = 0.20219 - 0.168 * log10(f_asph)

            if bullwinkle_fraction < 0.0:
                bullwinkle_fraction = 0.0
            elif bullwinkle_fraction > 0.303:
                bullwinkle_fraction = 0.303
        elif oil.api < 26.0:
            bullwinkle_fraction = 0.08
        elif oil.api > 50.0:
            bullwinkle_fraction = 0.303
        else:
            bullwinkle_fraction = -1.038 - 0.78935 * log10(1.0 / oil.api)

        bullwinkle_fraction = new_bull_calc(oil, bullwinkle_fraction)

    oil.bullwinkle_fraction = bullwinkle_fraction
    oil.estimated.bullwinkle_fraction = True


def new_bull_calc(imported_rec, bullwinkle_fraction):
    '''
        From the Adios2 c++ file OilInitialize.cpp, there is functionality
        inside the function CAdiosData::Bullwinkle() which is annotated
        in the code as 'new bull calc'.

        It uses the following definitions:
        - TG, Documented as the value 'dT/df - evaporation'.
              I can only assume this is the initial fractional rate of
              evaporation.
        - TBP, Documented as the 'ADIOS 1 liquid boiling point (bubble pt)'.
        - BullAdios1, which appears to be used to scale-average the initially
                      computed bullwinkle fraction.

        Regardless, in order to approximate what Adios2 is doing, we
        need this modification of our bullwinkle fraction.
    '''
    t_g = 1356.7 - 247.36 * log(imported_rec.api)
    t_bp = 532.98 - 3.1295 * imported_rec.api
    bull_adios1 = (483.0 - t_bp) / t_g

    if bull_adios1 < 0.0:
        bull_adios1 = 0.0
    elif bull_adios1 > 0.4:
        bull_adios1 = 0.4

    return 0.5 * (bullwinkle_fraction + bull_adios1)


def add_adhesion(imported_rec, oil):
    '''
        This is currently not used by the model, but we will get it
        if it exists.
        Otherwise, we will assign a constant.
    '''
    if imported_rec.adhesion is not None:
        oil.adhesion_kg_m_2 = imported_rec.adhesion
    else:
        oil.adhesion_kg_m_2 = 0.035
        oil.estimated.adhesion_kg_m_2 = True


def add_sulphur_mass_fraction(imported_rec, oil):
    '''
        This is currently not used by the model, but we will get it
        if it exists.
        Otherwise, we will assign a constant per the documentation.
    '''
    if imported_rec.sulphur is not None:
        oil.sulphur_fraction = imported_rec.sulphur
    else:
        oil.sulphur_fraction = 0.0
        oil.estimated.sulphur_fraction = True


def add_soluability(imported_rec, oil):
    '''
        There is no direct soluability attribute in the imported record,
        so we will just assign a constant per the documentation.
    '''
    oil.soluability = 0.0
    oil.estimated.soluability = True


def add_distillation_cut_boiling_point(imported_rec, oil):
    '''
        if cuts exist:
            copy them over
        else:
            get a single cut from the API
    '''
    for c in imported_rec.cuts:
        # Most of our oils seem to be fractional amounts regardless of
        # the stated cut units.  There are only a small number of outliers
        # - 2 cuts are negative, which is impossible
        # - 55 are between 1.0 and 10.0 which could possibly be percent
        #   values, but since they are so low, it is unlikely.
        if c.fraction >= 0.0 and c.fraction <= 1.0:
            oil.cuts.append(c)
        else:
            print ('{0}: {1}: bad distillation cut!'.format(imported_rec, c))

    if not oil.cuts:
        mass_left = 1.0

        mass_left -= sum([f.fraction for f in oil.sara_fractions
                          if f.sara_type in ('Resins', 'Asphaltenes')])
        # if imported_rec.resins:
        #     mass_left -= imported_rec.resins
        #
        # if imported_rec.asphaltene_content:
        #     mass_left -= imported_rec.asphaltene_content

        summed_boiling_points = []
        for t, f in get_boiling_points_from_api(5, mass_left, oil.api):
            added_to_sums = False

            for idx, [ut, summed_value] in enumerate(summed_boiling_points):
                if np.isclose(t, ut):
                    summed_boiling_points[idx][1] += f
                    added_to_sums = True
                    break

            if added_to_sums is False:
                summed_boiling_points.append([t, f])

        accumulated_frac = 0.0
        for t_i, fraction in summed_boiling_points:
            accumulated_frac += fraction
            oil.cuts.append(Cut(fraction=accumulated_frac, vapor_temp_k=t_i))

        oil.estimated.cuts = True


def add_molecular_weights(imported_rec, oil):
    '''
        Molecular weight units = g/mol
    '''
    for c in oil.cuts:
        mw_sat = get_saturate_molecular_weight(c.vapor_temp_k)
        if mw_sat is not None:
            mw = MolecularWeight(sara_type='Saturates',
                                 g_mol=mw_sat,
                                 ref_temp_k=c.vapor_temp_k)
            oil.molecular_weights.append(mw)

        mw_arom = get_aromatic_molecular_weight(c.vapor_temp_k)
        if mw_arom is not None:
            mw = MolecularWeight(sara_type='Aromatics',
                                 g_mol=mw_arom,
                                 ref_temp_k=c.vapor_temp_k)
            oil.molecular_weights.append(mw)


def get_saturate_molecular_weight(vapor_temp):
    '''
        (Reference: CPPF, eq. 2.48 and table 2.6)
        Note: T_b_inf is the upper cutoff temperature that we use.
              The implication is that a supposed saturate would need to
              have an infinite molecular weight to have a boiling point
              above the cutoff.  I don't know how realistic this is, but
              nevertheless, we don't calculate any distillation
              cuts above our cutoff.
    '''
    T_b_inf = 1070.0
    if vapor_temp < 1070.0:
        return (49.7 * (6.983 - log(T_b_inf - vapor_temp))) ** (3. / 2.)
    else:
        return None


def get_aromatic_molecular_weight(vapor_temp):
    '''
        (Reference: CPPF, eq. 2.48 and table 2.6)
        Note: T_b_inf is the upper cutoff temperature that we use.
              The implication is that a supposed saturate would need to
              have an infinite molecular weight to have a boiling point
              above the cutoff.  I don't know how realistic this is, but
              nevertheless, we don't calculate any distillation
              cuts above our cutoff.
    '''
    T_b_inf = 1015.0
    if vapor_temp < 1015.0:
        return (44.5 * (6.91 - log(T_b_inf - vapor_temp))) ** (3. / 2.)
    else:
        return None


def add_saturate_aromatic_fractions(imported_rec, oil):
    for f_sat, f_arom, T_i in get_sa_mass_fractions(oil):
        oil.sara_fractions.append(SARAFraction(sara_type='Saturates',
                                               fraction=f_sat,
                                               ref_temp_k=T_i))
        oil.sara_fractions.append(SARAFraction(sara_type='Aromatics',
                                               fraction=f_arom,
                                               ref_temp_k=T_i))


def adjust_molecular_weights(oil):
    '''
        Initially, we calculate our molecular weights based on the distillation
        cuts.  But after we have added all our saturate & aromatic fractions,
        there is a possibility that we have molecular weights calculated at
        temperatures where there exists no mass fraction.

        No mass, no molecular weight.  So we prune them.
    '''
    mass_temps = set([f.ref_temp_k for f in oil.sara_fractions])

    for idx, mw in reversed(list(enumerate(oil.molecular_weights))):
        if mw.ref_temp_k not in mass_temps:
            del oil.molecular_weights[idx]


def adjust_resin_asphaltene_fractions(imported_rec, oil):
    '''
        After we have added our saturate & aromatic fractions,
        the fraction sum may still be less than 1.0.

        If we have a resonable number of distillation cuts, we could
        make the rationalization that the remaining fraction is composed
        of resins and asphaltenes.
        We need to determine to a certain level of confidence if this is true,
        but if so we can scale up the resin and asphaltene amounts.
    '''
    sara_total = sum([sara.fraction for sara in oil.sara_fractions])
    if (not np.isclose(sara_total, 1.0) and sara_total < 1.0):
        # probably need a check to see if we had a reasonable number of
        # distillation cuts covering a reasonable range of temperatures.
        # ...or we could just not scale our fractions over a certain amount.
        ra_fraction = sum([sara.fraction
                           for sara in oil.sara_fractions
                           if sara.sara_type in ('Resins', 'Asphaltenes')])
        if ra_fraction > 0.0:
            sa_fraction = sum([sara.fraction
                               for sara in oil.sara_fractions
                               if sara.sara_type in ('Saturates',
                                                     'Aromatics')])
            leftover = 1.0 - sa_fraction
            scale = leftover / ra_fraction

            for sara in oil.sara_fractions:
                if sara.sara_type in ('Resins', 'Asphaltenes'):
                    sara.fraction *= scale

            print ('\tNew SARA total = {0}, RA fractions scaled by {1}'
                   .format(sum([sara.fraction for sara in oil.sara_fractions]),
                           scale))


def get_ptry_values(oil_obj, component_type, sub_fraction=None):
    '''
        This gives an initial trial estimate for each density component.

        In theory the fractionally weighted average of these densities,
        combined with the fractionally weighted average resin and asphaltene
        densities, should match the measured total oil density.

        :param oil_obj: an oil database object
        :param watson_factor: The characterization factor originally defined
                              by Watson et al. of the Universal Oil Products
                              in the mid 1930's
                              (Reference: CPPF, section 2.1.15 )
        :param sub_fraction: a list of fractions to be used in lieu of the
                             calculated cut fractions in the database.
    '''
    watson_factors = {'Saturates': 12, 'Aromatics': 10}
    watson_factor = watson_factors[component_type]

    previous_cut_fraction = 0.0
    for idx, c in enumerate(oil_obj.cuts):
        T_i = c.vapor_temp_k

        F_i = c.fraction - previous_cut_fraction
        previous_cut_fraction = c.fraction

        P_try = 1000 * (T_i ** (1.0 / 3.0) / watson_factor)

        if sub_fraction is not None:
            if len(sub_fraction) > idx:
                F_i = sub_fraction[idx]
            else:
                break

        yield (P_try, F_i, T_i, component_type)


def get_sa_mass_fractions(oil_obj):
    '''
        (A) if these hold true:
              - (i): oil library record contains summed mass fractions
                     (Saturate and aromatic combined)
                     or weight (%) for the distillation cuts
              - (ii): T(i) < 530K
            then:
              (Reference: CPPF, eq.s 3.77 and 3.78)
              - f(sat, i) = (fmass(i) *
                             (2.24 - 1.98 * SG(sat, i) - 0.009 * M(w, sat, i)))
              - if f(sat, i) >= fmass(i):
                  - f(sat, i) = fmass(i)
              - else if f(sat, i) < 0:
                  - f(sat, i) = 0
            else if these hold true:
              - (ii): T(i) >= 530K
            then:
              - f(sat, i) = fmass(i) / 2
        (B) else if there were no measured mass fractions in the imported
            record
              - apply (A) except fmass(i) = 1/5 for all cuts

        dependent on:
            - oil.molecular_weights[:].saturate
    '''
    fraction_sum = sum([sara.fraction for sara in oil_obj.sara_fractions
                        if sara.sara_type in ('Resins', 'Asphaltenes')])

    for P_try, F_i, T_i, c_type in get_ptry_values(oil_obj, 'Saturates'):
        if T_i < 530.0:
            sg = P_try / 1000
            mw = None
            for v in oil_obj.molecular_weights:
                if (np.isclose(v.ref_temp_k, T_i) and
                        v.sara_type == 'Saturates'):
                    mw = v.g_mol
                    break

            if mw is not None:
                f_sat = F_i * (2.2843 - 1.98138 * sg - 0.009108 * mw)

                if f_sat >= F_i:
                    f_sat = F_i
                elif f_sat < 0.0:
                    f_sat = 0.0

                f_arom = F_i - f_sat
            else:
                print '\tNo molecular weight at that temperature.'
                continue
        else:
            # Above 530K we will evenly split the saturates & aromatics
            f_sat = f_arom = F_i / 2

        if fraction_sum >= 1.0:
            # print 'we cannot add any more'
            break
        elif f_sat + f_arom + fraction_sum > 1.0:
            # we need to scale our sub-fractions so that the
            # fraction sum adds up to 1.0
            scale = (1.0 - fraction_sum) / (f_sat + f_arom)
            f_sat *= scale
            f_arom *= scale

            fraction_sum += f_sat + f_arom
            yield (f_sat, f_arom, T_i)
        else:
            fraction_sum += f_sat + f_arom
            yield (f_sat, f_arom, T_i)


def add_component_densities(imported_rec, oil):
    '''
        (Reference: CPPF, eq. 2.13 and table 9.6)
        dependent on:
        - P_0_oil: oil density at 15C (estimation 1)
        - fmass_0_j: saturate & aromatic mass fractions (estimation 14,15)
    '''
    oil.sara_densities.append(SARADensity(sara_type='Asphaltenes',
                                          density=1100.0,
                                          ref_temp_k=1015.0))
    oil.sara_densities.append(SARADensity(sara_type='Resins',
                                          density=1100.0,
                                          ref_temp_k=1015.0))

    sa_ratios = list(get_sa_mass_fractions(oil))

    ptry_values = (list(get_ptry_values(oil, 'Saturates',
                                        [r[0] for r in sa_ratios])) +
                   list(get_ptry_values(oil, 'Aromatics',
                                        [r[1] for r in sa_ratios])))

    ra_ptry_values = [(1100.0, f.fraction)
                      for f in oil.sara_fractions
                      if f.sara_type in ('Resins', 'Asphaltenes')]

    ptry_avg_density = sum([(P_try * F_i)
                            for P_try, F_i, T_i, c_type in ptry_values] +
                           [(P_try * F_i) for P_try, F_i in ra_ptry_values])

    total_sa_fraction = sum([F_i for P_try, F_i, T_i, c_type in ptry_values])

    total_ra_fraction = sum([f.fraction for f in oil.sara_fractions
                             if f.sara_type in ('Resins', 'Asphaltenes')])
    oil_density = density_at_temperature(oil, 288.15)

    # print ('\n\nNow we will try to adjust our ptry densities '
    #        'to match the oil total density')
    oil_sa_avg_density = ((oil_density - total_ra_fraction * 1100.0) /
                          total_sa_fraction)

    density_adjustment = oil_sa_avg_density / ptry_avg_density
    ptry_values = [(P_try * density_adjustment, F_i, T_i, c_type)
                   for P_try, F_i, T_i, c_type in ptry_values]

    for P_try, F_i, T_i, c_type in ptry_values:
        oil.sara_densities.append(SARADensity(sara_type=c_type,
                                              density=P_try,
                                              ref_temp_k=T_i))


def add_k0y(imported_rec, oil):
    if imported_rec.k0y is not None:
        oil.k0y = imported_rec.k0y
    else:
        oil.k0y = 2.02e-06


def reject_oil_if_bad(imported_rec, oil):
    '''
        Here, we have an oil in which all estimations have been made.
        We will now check the imported record and the oil object to see
        if there are any detectable flaws.
        If any flaw is detected, we will raise the OilRejected exception.
        All flaws will be compiled into a list of error messages to be passed
        into the exception.
    '''
    errors = []

    if imported_rec_was_manually_rejected(imported_rec):
        errors.append('Imported Record was manually rejected')

    if not oil_has_kvis(oil):
        errors.append('Oil has no kinematic viscosities')

    if oil_has_duplicate_cuts(oil):
        errors.append('Oil has duplicate cuts')

    if oil_has_heavy_sa_components(oil):
        errors.append('Oil has heavy SA components')

    if not oil_api_matches_density(oil):
        errors.append('Oil API does not match its density')

    if errors:
        raise OilRejected(errors, imported_rec.adios_oil_id)


def imported_rec_was_manually_rejected(imported_rec):
    '''
        This list was initially compiled to try and fix some anomalies
        that were showing up in the oil query form.

        When we update the source file that provides our imported record
        data, we should revisit this list.
        We should also revisit this list as we add methods to detect flaws
        in our oil record.
    '''
    adios_oil_id = imported_rec.adios_oil_id
    if adios_oil_id in ('AD00121',  # BCF 13
                        'AD02042',  # BOSCAN
                        'AD02130',  # FOROOZAN
                        'AD02240',  # LUCULA
                        ):
        return True
    return False


def oil_has_kvis(oil):
    '''
        Our oil record should have at least one kinematic viscosity when
        estimations are complete.
    '''
    if len(oil.kvis) > 0:
        return True
    else:
        return False


def oil_has_duplicate_cuts(oil):
    '''
        Some oil records have been found to have distillation cuts with
        duplicate vapor temperatures, and the fraction that should be chosen
        at that temperature is ambiguous.
    '''
    unique_temps = set([o.vapor_temp_k for o in oil.cuts])

    if len(oil.cuts) != len(unique_temps):
        return True
    else:
        return False


def oil_has_heavy_sa_components(oil):
    '''
        Some oil records have been found to have Saturate & Asphaltene
        densities that were calculated to be heavier than the Resins &
        Asphaltenes.
        This is highly improbable and indicates the record has problems
        with its imported data values.
    '''
    for d in oil.sara_densities:
        if d.sara_type in ('Saturates', 'Aromatics'):
            if d.density > 1100.0:
                return True

    return False


def oil_api_matches_density(oil):
    '''
        The oil API should pretty closely match its density at 15C.
    '''
    d_0 = density_at_temperature(oil, 273.15 + 15)
    api_from_density = (141.5 * 1000 / d_0) - 131.5

    if np.isclose(oil.api, api_from_density, atol=1.0):
        return True

    print '(oil.api, api_from_density) = ', (oil.api, api_from_density)
    return False
