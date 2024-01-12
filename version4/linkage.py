import random
import argparse
import pysam

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate generations of families and populations")
    parser.add_argument("vcf", help="Input VCF file with initial individuals")
    parser.add_argument("output_vcf_prefix", help="Prefix for output VCF files for each population")
    parser.add_argument("output_ped_prefix", help="Prefix for output PED files for each population")
    parser.add_argument("-g", "--num_generations", type=int, default=3, help="Number of generations to simulate (default: 3)")
    parser.add_argument("-p", "--num_populations", type=int, default=3, help="Number of populations to simulate (default: 3)")
    parser.add_argument("-c", "--num_couples", type=int, default=20, help="Number of initial couples (default: 20)")
    parser.add_argument("-k", "--num_non_recombining_positions", type=int, default=10, help="Number of consecutive non-recombining positions (default: 1000)")
    
    return parser.parse_args()

def read_vcf(filename):
    '''读取输入VCF文件,获取样本列表(samples)、原始变异位点列表(variants)和输入VCF头信息 (header)'''
    vcf = pysam.VariantFile(filename)  
    samples = list(vcf.header.samples)  # 样本列表
    variants = list(vcf.fetch())        # 变异位点列表
    return samples, variants, vcf.header

def write_vcf(out_vcf, new_header, new_variants):
    '''将生成的新变异位点写入输出的VCF文件。'''
    out_vcf =  pysam.VariantFile(out_vcf, "w", header=new_header)
    for variant in new_variants:
        out_vcf.write(variant)
    out_vcf.close()

def write_ped(filename, trios):
    '''将生成的三代家庭关系写入输出的PED文件'''
    with open(filename, 'w') as f:
        f.write('child\tfather\tmother\n')
        for k, v in trios.items():
            f.write(f"{k}\t{v[0]}\t{v[1]}\n")

def generate_recombination_intervals(num_recombinations, num_variants, non_recombining_positions):
    recombination_points = sorted(random.sample(set(range(1, num_variants)) - set(non_recombining_positions), 2 * num_recombinations))
    recombination_intervals = [(recombination_points[i], recombination_points[i + 1]) for i in range(0, len(recombination_points), 2)]
    return recombination_intervals

def is_variant_in_recombination_intervals(variant_pos, recombination_intervals):
    '''检查一个变异位点是否在给定的重组区间内'''
    for start, end in recombination_intervals:
        if start <= variant_pos <= end:
            return True
    return False

def generate_non_recombining_positions(chromosome_length, num_non_recombining_positions):
    '''生成不参与重组的位置'''
    start_position = random.randint(1, chromosome_length - num_non_recombining_positions)
    return list(range(start_position, start_position + num_non_recombining_positions))

def simulate_population(vcf_filename, output_vcf_prefix, output_ped_prefix, num_generations, num_couples, num_non_recombining_positions, num_populations, num_recombinations):
    '''模拟多个群体的演化'''
    samples, variants, header = read_vcf(vcf_filename)
    chromosome_length = max(variant.pos for variant in variants)

    for population_num in range(1, num_populations + 1):
        population_samples = []
        trios = {}

        for couple_num in range(1, num_couples + 1):
            father = f"pop{population_num}_father{couple_num}"
            mother = f"pop{population_num}_mother{couple_num}"
            child1 = f"pop{population_num}_child{couple_num}_1"
            child2 = f"pop{population_num}_child{couple_num}_2"

            population_samples.extend([father, mother, child1, child2])

            trios[child1] = (father, mother, child1)
            trios[child2] = (father, mother, child2)

        non_recombining_positions = generate_non_recombining_positions(chromosome_length, num_non_recombining_positions)

        for generation_num in range(1, num_generations + 1):
            new_variants, new_header = simulate_trio(num_recombinations, population_samples, variants, header, trios, non_recombining_positions)
            output_vcf_filename = f"{output_vcf_prefix}_pop{population_num}_gen{generation_num}.vcf"
            output_ped_filename = f"{output_ped_prefix}_pop{population_num}_gen{generation_num}.ped"

            write_vcf(output_vcf_filename, new_header, new_variants)
            write_ped(output_ped_filename, trios)

            # Update for the next generation
            samples, variants, header = read_vcf(output_vcf_filename)

def simulate_trio(num_recombinations, out_samples, variants, out_header, trios, non_recombining_positions):
    recombination_intervals = generate_recombination_intervals(num_recombinations, len(variants), non_recombining_positions)
    new_variants = []

    for variant in variants:
        new_variant = out_header.new_record()
        new_variant.chrom = variant.chrom
        new_variant.pos = variant.pos
        new_variant.ref = variant.ref
        new_variant.alts = variant.alts
        new_variant.id = variant.id
        new_variant.qual = variant.qual

        for key, value in variant.info.items():
            new_variant.info[key] = value

        for sample in out_samples:
            if "child" in sample:
                gt1 = variant.samples[trios[sample][0]]['GT']
                gt2 = variant.samples[trios[sample][1]]['GT']

                if is_variant_in_recombination_intervals(variant.pos, recombination_intervals):
                    gt1 = gt1[::-1]
                    gt2 = gt2[::-1]

                child_gt = (gt1[0], gt2[0])
                new_variant.samples[sample]['GT'] = child_gt
                new_variant.samples[sample].phased = True
            else:
                new_variant.samples[sample]['GT'] = variant.samples[sample]['GT']
                new_variant.samples[sample].phased = True

        new_variants.append(new_variant)

    return new_variants, out_header

# Rest of the code remains the same...

def main():
    args = parse_args()

    for population_num in range(1, args.num_populations + 1):
        input_vcf_filename = args.vcf if population_num == 1 else f"{args.output_vcf_prefix}_pop{population_num - 1}_gen{args.num_generations}.vcf"
        output_vcf_prefix = f"{args.output_vcf_prefix}_pop{population_num}"
        output_ped_prefix = f"{args.output_ped_prefix}_pop{population_num}"

        simulate_population(input_vcf_filename, output_vcf_prefix, output_ped_prefix, args.num_generations, args.num_couples, args.num_non_recombining_positions, args.num_populations, args.num_recombinations)

if __name__ == "__main__":
    main()
